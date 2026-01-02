# ==============================================================================
#  lingo_aura_standalone.py (CMU-MOSEI 最终版)
# ==============================================================================
#
# HOW TO RUN:
# 1. 确保已根据 requirements.txt 安装所有依赖 (torch, transformers, pandas, etc.)
# 2. 确保 `mmsdk` 已在您的环境中正确安装 (用于获取数据集划分ID)。
# 3. 手动从Hugging Face下载所有必需的CMU-MOSEI .csd文件，并放入 `data/cmumosei/` 文件夹。
#    必需文件:
#    - CMU_MOSEI_VisualFacet42.csd
#    - CMU_MOSEI_COVAREP.csd
#    - CMU_MOSEI_TimestampedWords.csd
#    - CMU_MOSEI_Labels.csd
# 4. 确保 `prompts/cognitive_informed_prompt.txt` 文件已创建并填充内容。
# 5. (可选) 运行独立的认知标签生成脚本，或让本脚本自动生成一个【模拟】版本。
# 6. 直接运行: python lingo_aura_standalone.py

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 导入 mmsdk 仅仅是为了使用它的数据集划分(standard_folds)
from mmsdk import mmdatasdk as md

# --- 1. 全局配置 ---
class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    OUTPUT_DIR = "output/all_model_LoRA_attention_right_label_r16_normalize_visionpre"

    LLM_NAME = "./phi-2"
    VISUAL_FEATURE_DIM = 35      # CMU_MOSEI_VisualFacet42 的特征维度
    ACOUSTIC_FEATURE_DIM = 74    # CMU_MOSEI_COVAREP 的特征维度

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    # LEARNING_RATE = 1e-4
    # GRADIENT_ACCUMULATION_STEPS = 8 # <-- 我们累积8个小批次，等效于 batch_size=8

# --- 3. 数据加载与处理 ---
class MOSEIDataset(Dataset):
    # ✨✨✨ 1. 修改 __init__ ✨✨✨
    def __init__(self, cognitive_df, split_ids, prompt_template, 
                 visual_stats=None, acoustic_stats=None, is_train=False):
        self.cognitive_df = cognitive_df.set_index('segment_id')
        self.split_ids = set(split_ids)
        self.prompt_template = prompt_template
        
        self.visual_field = 'CMU_MOSEI_VisualFacet42'
        self.acoustic_field = 'CMU_MOSEI_COVAREP'
        self.label_field = 'CMU_MOSEI_Labels'
        
        self.is_train = is_train

        print("正在手动加载 .csd 文件...")
        self.features = self._load_csd_files()

        print("正在对齐特征...")
        self.aligned_features = self._align_features()
        
        self.raw_data  = self._prepare_data()#原始数据

        # 如果是训练集，则计算并存储统计量
        if self.is_train:
            self.visual_mean, self.visual_std = self._compute_stats([d['visual'] for d in self.raw_data])
            self.acoustic_mean, self.acoustic_std = self._compute_stats([d['acoustic'] for d in self.raw_data])
        # 如果是验证集，则使用传入的统计量
        elif visual_stats and acoustic_stats:
            self.visual_mean, self.visual_std = visual_stats
            self.acoustic_mean, self.acoustic_std = acoustic_stats
        else:
            # 如果是验证集但没有传入统计量，这是一个错误
            raise ValueError("Validation dataset requires stats from the training set.")

        # ✨✨✨ 现在应用归一化 ✨✨✨
        self.data = self._normalize_data(self.raw_data)


    def _compute_stats(self, features_list):
        # 辅助函数，用于计算均值和标准差
        if not features_list:
            # 处理空数据集的情况
            return torch.tensor(0.0), torch.tensor(1.0)
        
        all_features = torch.cat(features_list, dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        # 防止标准差为0
        std[std == 0] = 1.0
        return mean, std
        
    def _normalize_data(self, raw_data_list):
        normalized_data = []
        for item in raw_data_list:
            normalized_item = item.copy() # 浅拷贝
            normalized_item['visual'] = (item['visual'] - self.visual_mean) / self.visual_std
            normalized_item['acoustic'] = (item['acoustic'] - self.acoustic_mean) / self.acoustic_std
            normalized_data.append(normalized_item)
        return normalized_data
    
    
    def _load_csd_files(self):
        recipe = {
            self.visual_field: os.path.join(Config.DATA_PATH, self.visual_field + '.csd'),
            self.acoustic_field: os.path.join(Config.DATA_PATH, self.acoustic_field + '.csd'),
            self.label_field: os.path.join(Config.DATA_PATH, self.label_field + '.csd'),
        }
        temp_dataset = md.mmdataset(recipe)
        features = {
            'visual': temp_dataset[self.visual_field].data,
            'acoustic': temp_dataset[self.acoustic_field].data,
            'labels': temp_dataset[self.label_field].data,
        }
        print("手动加载 .csd 文件完成。")
        return features

    def _align_features(self):
        aligned = {}
        for seg_id, label_data in self.features['labels'].items():
            if seg_id in self.features['visual'] and seg_id in self.features['acoustic']:
                vis_intervals, vis_feats = self.features['visual'][seg_id]['intervals'], self.features['visual'][seg_id]['features']
                acou_intervals, acou_feats = self.features['acoustic'][seg_id]['intervals'], self.features['acoustic'][seg_id]['features']
                
                # MOSEI的标签是一个时间点，不是区间，我们简单取其前后0.5秒
                time_point = label_data['intervals'][0][0]
                start, end = max(0, time_point - 0.5), time_point + 0.5

                vis_mask = (vis_intervals[:, 0] >= start) & (vis_intervals[:, 1] <= end)
                acou_mask = (acou_intervals[:, 0] >= start) & (acou_intervals[:, 1] <= end)
                
                if np.any(vis_mask) and np.any(acou_mask):
                    aligned[seg_id] = {
                        'visual': vis_feats[vis_mask],
                        'acoustic': acou_feats[acou_mask]
                    }
        return aligned

    def _prepare_data(self):
        prepared_data = []
        skipped_count = 0 # 添加一个计数器用于统计跳过的样本
        for segment_id in self.aligned_features.keys():
            if segment_id not in self.split_ids: continue
            try:
                visual_raw = self.aligned_features[segment_id]['visual']
                acoustic_raw = self.aligned_features[segment_id]['acoustic']
                
                # ✨✨✨【核心修复与检查】✨✨✨
                # 检查1：确保对齐后的特征张量不是空的
                if visual_raw.shape[0] == 0 or acoustic_raw.shape[0] == 0:
                    skipped_count += 1
                    continue # 如果任一特征为空，则直接跳过这个样本

                visual = visual_raw.astype(np.float32)
                acoustic = acoustic_raw.astype(np.float32)

                # 检查2 (可选但推荐): 确保原始数据不包含nan或inf
                if np.isnan(visual).any() or np.isinf(visual).any() or \
                np.isnan(acoustic).any() or np.isinf(acoustic).any():
                    skipped_count += 1
                    continue # 如果包含非法值，也跳过

                text = self.cognitive_df.loc[segment_id]['text']
                emotion_score = self.cognitive_df.loc[segment_id]['emotion_score']
                cognitive_label = json.loads(self.cognitive_df.loc[segment_id]['cognitive_label'])
                human_prompt = self.prompt_template.split("### Assistant:")[0].format(
                    information_stance=cognitive_label.get("Information Stance", "N/A"),
                    reasoning_mode=cognitive_label.get("Reasoning Mode", "N/A"),
                    transcription=text)
                assistant_response = self.prompt_template.split("### Assistant:")[1].format(emotion_score=emotion_score)
                # prepared_data.append({
                #     'visual': torch.from_numpy(visual),
                #     'acoustic': torch.from_numpy(acoustic),
                #     'full_text': human_prompt + assistant_response,
                #     'prompt_len': len(human_prompt)
                # })
                ####
                prepared_data.append({
                'visual': torch.from_numpy(visual),
                'acoustic': torch.from_numpy(acoustic),
                'text': self.cognitive_df.loc[segment_id]['text'],
                'emotion_score': self.cognitive_df.loc[segment_id]['emotion_score'],
                'cognitive_label': cognitive_label,
                    })
            except Exception: 
                skipped_count += 1 # 捕获其他可能的异常并跳过
                continue
        if skipped_count > 0:
            print(f"警告: 在数据准备过程中，由于特征为空或包含非法值，共跳过了 {skipped_count} 个样本。")
        return prepared_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- 4. 模型架构 ---
class LingoAuraLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ###量化配置
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_NAME,
            quantization_config=quant_config, # <-- 明确指定使用 float16 半精度
            device_map=config.DEVICE,
            trust_remote_code=True
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        # --- d. 定义LoRA配置 ---
        # 对于 Phi-2, 常见的 target_modules 是 "q_proj", "k_proj", "v_proj", "dense"
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "dense"], 
            lora_dropout=0.1,
            bias="none",
        )
        # --- e. 将LoRA适配器应用到LLM上 ---
        self.llm = get_peft_model(self.llm, lora_config)

        # self.llm = AutoModelForCausalLM.from_pretrained(config.LLM_NAME, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
        llama_hidden_size = self.llm.config.hidden_size
        self.visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
        self.acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)
       
       
        # ✨✨✨【核心修改 1】: 定义跨模态注意力模块 ✨✨✨
        # 我们使用一个简单的单头注意力，头数(num_heads)可以调整
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=llama_hidden_size, num_heads=4, batch_first=True
        )
        self.acoustic_attention = nn.MultiheadAttention(
            embed_dim=llama_hidden_size, num_heads=4, batch_first=True
        )

        # 我们让projectors跟随LLM的设备
        llm_device = self.llm.device
        
        # 3. 将 projectors 移动到与LLM相同的设备上
        self.visual_projector.to(llm_device, dtype=torch.bfloat16)
        self.acoustic_projector.to(llm_device, dtype=torch.bfloat16)

        # 将注意力模块也移动到正确的设备和类型
        self.visual_attention.to(llm_device, dtype=torch.bfloat16)
        self.acoustic_attention.to(llm_device, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels, visual_features, acoustic_features):
        projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
        projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
        # visual_token_embeds = projected_visual.mean(dim=1, keepdim=True)
        # acoustic_token_embeds = projected_acoustic.mean(dim=1, keepdim=True)

        # 2. 获取文本嵌入 (和以前一样)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # ✨✨✨【核心修改 2】: 使用注意力动态融合，而不是平均池化 ✨✨✨
        # 我们用文本的第一个token ([CLS] 或起始符) 的嵌入作为“查询(query)”
        # 这个 query 代表了整句话的语义概括
        query_embed = text_embeds[:, 0:1, :].to(torch.bfloat16) # 形状: (batch, 1, hidden_size)

        # 视觉融合: query去查询 projected_visual 序列
        # attn_output 形状: (batch, 1, hidden_size)
        visual_token_embeds, _ = self.visual_attention(
            query=query_embed, key=projected_visual, value=projected_visual
        )
        print("视觉注意力权重形状:", _ .shape) 

        # 听觉融合: query去查询 projected_acoustic 序列
        acoustic_token_embeds, _ = self.acoustic_attention(
            query=query_embed, key=projected_acoustic, value=projected_acoustic
        )

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        inputs_embeds = torch.cat([visual_token_embeds, acoustic_token_embeds, text_embeds], dim=1)
        extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        final_attn_mask = torch.cat([extra_tokens_mask, attention_mask], dim=1)
        extra_labels = torch.full((labels.shape[0], 2), -100, device=labels.device)
        final_labels = torch.cat([extra_labels, labels], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=final_attn_mask, labels=final_labels, return_dict=True)
        return outputs.loss

def create_training_collate_fn(tokenizer, prompt_template):
    def collate_fn(batch):
        # 1. 准备多模态特征 (保持不变)
        visual_features = pad_sequence([item['visual'] for item in batch], batch_first=True, padding_value=0.0)
        acoustic_features = pad_sequence([item['acoustic'] for item in batch], batch_first=True, padding_value=0.0)

        # 2. 准备文本、输入和标签
        human_prompts_full = []
        assistant_responses = []

        human_template, assistant_template = prompt_template.split("### Assistant:")
        human_template += "### Assistant:"

        for item in batch:
            # 这里的 item['cognitive_label'] 和 item['text'] 需要从 batch 中获取
            # 这取决于您的 Dataset __getitem__ 的实现，需要确保它们被正确传递
            # 我们假设您的 Dataset __getitem__ 返回一个包含所有需要信息的字典
            human_prompts_full.append(
                human_template.format(
                    information_stance=item['cognitive_label'].get("Information Stance", "N/A"),
                    reasoning_mode=item['cognitive_label'].get("Reasoning Mode", "N/A"),
                    transcription=item['text']
                )
            )
            assistant_responses.append(
                assistant_template.format(emotion_score=item['emotion_score'])
            )

        # 3. 为模型输入进行分词 (Human Prompt + Assistant Response)
        full_texts = [p + r + tokenizer.eos_token for p, r in zip(human_prompts_full, assistant_responses)]
        inputs_tokenized = tokenizer(
            full_texts, padding='longest', return_tensors="pt", truncation=True, max_length=512
        )

        # 4. 创建 labels
        labels = inputs_tokenized['input_ids'].clone()

        # 5. 单独分词 Human Prompt 以获取其长度
        prompts_tokenized = tokenizer(
            human_prompts_full, padding='longest', return_tensors="pt", truncation=True, max_length=512
        )
        prompt_lengths = torch.sum(prompts_tokenized.attention_mask, dim=1)

        # 6. 精确屏蔽
        for i in range(len(batch)):
            labels[i, :prompt_lengths[i]] = -100 # 屏蔽 prompt
            
            score_match = re.search(r"[-+]?\d+(?:\.\d+)?", assistant_responses[i])
            if score_match:
                score_str = score_match.group(0)
                assistant_prefix = assistant_template.split("{")[0]
                prefix_tokens = tokenizer(assistant_prefix, add_special_tokens=False)['input_ids']
                score_tokens = tokenizer(score_str, add_special_tokens=False)['input_ids']
                
                start_of_score = prompt_lengths[i] + len(prefix_tokens)
                end_of_score = start_of_score + len(score_tokens)
                
                labels[i, prompt_lengths[i]:start_of_score] = -100
                labels[i, end_of_score:] = -100
        
        # 注意：返回的字典需要和您的 Dataset __getitem__ 对应
        return {
            'input_ids': inputs_tokenized['input_ids'],
            'attention_mask': inputs_tokenized['attention_mask'],
            'labels': labels,
            'visual_features': visual_features,
            'acoustic_features': acoustic_features,
        }
    return collate_fn
# --- 6. 主执行函数 ---
def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*20 + " 1. 准备认知标签 " + "="*20)
    # if not os.path.exists(config.COGNITIVE_LABELS_CSV):
    #     temp_recipe = {
    #         'CMU_MOSEI_TimestampedWords': os.path.join(config.DATA_PATH, 'CMU_MOSEI_TimestampedWords.csd'),
    #         'CMU_MOSEI_Labels': os.path.join(config.DATA_PATH, 'CMU_MOSEI_Labels.csd')
    #     }
    #     temp_dataset = md.mmdataset(temp_recipe)
    #     generate_mock_cognitive_labels(temp_dataset, config.COGNITIVE_LABELS_CSV)
    
    print(f"从 '{config.COGNITIVE_LABELS_CSV}' 加载认知标签。")
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()


    print("\n" + "="*20 + " 2. 初始化模型 " + "="*20)
    model = LingoAuraLLM(config)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE,weight_decay=0.01 )



    print("\n" + "="*20 + " 3. 构建数据集 " + "="*20)
    train_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_train_fold, 
                                 prompt_template, is_train=True)    
    # b. 提取训练集的统计量
    visual_stats = (train_dataset.visual_mean, train_dataset.visual_std)
    acoustic_stats = (train_dataset.acoustic_mean, train_dataset.acoustic_std)

    # c. 将训练集的统计量传入验证集
    val_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_valid_fold, 
                               prompt_template, visual_stats=visual_stats, acoustic_stats=acoustic_stats)
    

    collate_fn = create_training_collate_fn(model.tokenizer, prompt_template)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=16,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=16,pin_memory=True)
    

    print("\n" + "="*20 + " 4. 开始训练 " + "="*20)
    # --- ✨✨✨ 核心修改 1: 初始化用于追踪最佳模型的变量 ✨✨✨ ---
    best_val_loss = float('inf') # 将初始最佳损失设为无穷大
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
        
        optimizer.zero_grad() 

        for i, batch in enumerate(progress_bar):
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            loss = model(**batch)
            # loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # 累加未缩放的损失用于显示
            # current_loss = loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            current_loss = loss.item() 
            train_loss += current_loss
            progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
            
            # if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # --- ✨✨✨ 核心修改：在更新前进行梯度裁剪 ✨✨✨ ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / len(train_loader.dataset) * config.BATCH_SIZE # 更精确的损失计算
        print(f"Epoch {epoch+1} - 平均训练损失: {avg_train_loss:.4f}")

        # --- 验证循环 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validating]"):
                batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
                loss = model(**batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - 验证损失: {avg_val_loss:.4f}")
        
        # --- ✨✨✨ 核心修改 2: 检查并保存最佳模型 ✨✨✨ ---
        if avg_val_loss < best_val_loss:
            print(f"发现新的最佳模型！验证损失从 {best_val_loss:.4f} 降低到 {avg_val_loss:.4f}。正在保存...")
            best_val_loss = avg_val_loss
            
            # --- 保存模型的逻辑 ---
            save_path = config.OUTPUT_DIR
            os.makedirs(save_path, exist_ok=True)
            
            # a. 保存 LoRA 适配器权重
            model.llm.save_pretrained(save_path)
            
            # b. 保存 tokenizer
            model.tokenizer.save_pretrained(save_path)
            
            # c. 保存可训练的 Projector 权重
            torch.save(model.visual_projector.state_dict(), os.path.join(save_path, "visual_projector.pt"))
            torch.save(model.acoustic_projector.state_dict(), os.path.join(save_path, "acoustic_projector.pt"))

            # ✨✨✨【关键补充】: 保存新的 Attention 模块的权重 ✨✨✨
            print(" - 正在保存注意力模块权重...")
            torch.save(model.visual_attention.state_dict(), os.path.join(save_path, "visual_attention.pt"))
            torch.save(model.acoustic_attention.state_dict(), os.path.join(save_path, "acoustic_attention.pt"))
            
            print(f"最佳模型已保存至: {save_path}")


    # ✨✨✨ 3. 训练结束后，保存统计量 ✨✨✨
    stats_to_save = {
        'visual_mean': visual_stats[0].tolist(),
        'visual_std': visual_stats[1].tolist(),
        'acoustic_mean': acoustic_stats[0].tolist(),
        'acoustic_std': acoustic_stats[1].tolist(),
    }
    with open(os.path.join(config.OUTPUT_DIR, 'normalization_stats.json'), 'w') as f:
        json.dump(stats_to_save, f)
    
    print(f"归一化统计量已保存至: {config.OUTPUT_DIR}")


    print("\n" + "="*60)
    print("所有训练 Epoch 完成！")
    print(f"最终保存在 '{config.OUTPUT_DIR}' 的是验证集上表现最佳的模型。")
    print(f"最佳验证损失为: {best_val_loss:.4f}")
    print("="*60)
if __name__ == "__main__":
    # 确保所有文件路径正确后再取消注释运行
    main()