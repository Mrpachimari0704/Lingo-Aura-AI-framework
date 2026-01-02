# ==============================================================================
#  lingo_aura_standalone_ddp.py (CMU-MOSEI 多显卡 DDP 版 - Updated Logic)
# ==============================================================================

import os
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse 

import torch
import torch.nn as nn
import torch.nn.functional as F # 添加 functional
import torch.distributed as dist 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from mmsdk import mmdatasdk as md

# --- 1. 全局配置 ---
class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    OUTPUT_DIR = "output/all_model_LoRA_doublemlp_meanpool_contrastive_ddp" # 更新输出目录名

    LLM_NAME = "./Mistral-7B-Instruct-v0.2"
    VISUAL_FEATURE_DIM = 35
    ACOUSTIC_FEATURE_DIM = 74

    EPOCHS = 20
    BATCH_SIZE = 10  # 单卡 Batch Size
    # LEARNING_RATE = 5e-5

# --- 2. DDP 辅助函数 ---
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return local_rank, rank, world_size
    else:
        print("未检测到 DDP 环境，回退到单卡模式。")
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# --- 3. 数据加载 (保持不变) ---
class MOSEIDataset(Dataset):
    def __init__(self, cognitive_df, split_ids, prompt_template, 
                 visual_stats=None, acoustic_stats=None, is_train=False):
        self.cognitive_df = cognitive_df.set_index('segment_id')
        self.split_ids = set(split_ids)
        self.prompt_template = prompt_template
        self.visual_field = 'CMU_MOSEI_VisualFacet42'
        self.acoustic_field = 'CMU_MOSEI_COVAREP'
        self.label_field = 'CMU_MOSEI_Labels'
        self.is_train = is_train

        if is_main_process(): print("正在手动加载 .csd 文件...")
        self.features = self._load_csd_files()
        if is_main_process(): print("正在对齐特征...")
        self.aligned_features = self._align_features()
        self.raw_data  = self._prepare_data()

        if self.is_train:
            self.visual_mean, self.visual_std = self._compute_stats([d['visual'] for d in self.raw_data])
            self.acoustic_mean, self.acoustic_std = self._compute_stats([d['acoustic'] for d in self.raw_data])
        elif visual_stats and acoustic_stats:
            self.visual_mean, self.visual_std = visual_stats
            self.acoustic_mean, self.acoustic_std = acoustic_stats
        else:
            raise ValueError("Validation dataset requires stats from the training set.")
        self.data = self._normalize_data(self.raw_data)

    def _compute_stats(self, features_list):
        if not features_list: return torch.tensor(0.0), torch.tensor(1.0)
        all_features = torch.cat(features_list, dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        std[std == 0] = 1.0
        return mean, std
        
    def _normalize_data(self, raw_data_list):
        normalized_data = []
        for item in raw_data_list:
            normalized_item = item.copy()
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
        return {
            'visual': temp_dataset[self.visual_field].data,
            'acoustic': temp_dataset[self.acoustic_field].data,
            'labels': temp_dataset[self.label_field].data,
        }

    def _align_features(self):
        aligned = {}
        for seg_id, label_data in self.features['labels'].items():
            if seg_id in self.features['visual'] and seg_id in self.features['acoustic']:
                vis_intervals, vis_feats = self.features['visual'][seg_id]['intervals'], self.features['visual'][seg_id]['features']
                acou_intervals, acou_feats = self.features['acoustic'][seg_id]['intervals'], self.features['acoustic'][seg_id]['features']
                time_point = label_data['intervals'][0][0]
                start, end = max(0, time_point - 0.5), time_point + 0.5
                vis_mask = (vis_intervals[:, 0] >= start) & (vis_intervals[:, 1] <= end)
                acou_mask = (acou_intervals[:, 0] >= start) & (acou_intervals[:, 1] <= end)
                if np.any(vis_mask) and np.any(acou_mask):
                    aligned[seg_id] = {'visual': vis_feats[vis_mask], 'acoustic': acou_feats[acou_mask]}
        return aligned

    def _prepare_data(self):
        prepared_data = []
        for segment_id in self.aligned_features.keys():
            if segment_id not in self.split_ids: continue
            try:
                visual_raw = self.aligned_features[segment_id]['visual']
                acoustic_raw = self.aligned_features[segment_id]['acoustic']
                if visual_raw.shape[0] == 0 or acoustic_raw.shape[0] == 0: continue
                visual = visual_raw.astype(np.float32)
                acoustic = acoustic_raw.astype(np.float32)
                if np.isnan(visual).any() or np.isinf(visual).any() or np.isnan(acoustic).any() or np.isinf(acoustic).any(): continue
                cognitive_label = json.loads(self.cognitive_df.loc[segment_id]['cognitive_label'])
                prepared_data.append({
                    'visual': torch.from_numpy(visual),
                    'acoustic': torch.from_numpy(acoustic),
                    'text': self.cognitive_df.loc[segment_id]['text'],
                    'emotion_score': self.cognitive_df.loc[segment_id]['emotion_score'],
                    'cognitive_label': cognitive_label,
                })
            except Exception: continue
        return prepared_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- 4. 模型架构 (✨✨✨ 已按照您的要求完全修改 ✨✨✨) ---
class LingoAuraLLM(nn.Module):
    def __init__(self, config, local_rank):
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_NAME,
            quantization_config=quant_config, 
            device_map={"": local_rank},
            trust_remote_code=True,
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  
            lora_dropout=0.05,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_config)

        llama_hidden_size = self.llm.config.hidden_size
        
        # ✨✨✨ 1. 双层 MLP 投影 (Double MLP) ✨✨✨
        self.visual_projector = nn.Sequential(
            nn.Linear(config.VISUAL_FEATURE_DIM, config.VISUAL_FEATURE_DIM * 2),
            nn.ReLU(),
            nn.Linear(config.VISUAL_FEATURE_DIM * 2, llama_hidden_size),
            nn.LayerNorm(llama_hidden_size)
        )
        self.acoustic_projector = nn.Sequential(
            nn.Linear(config.ACOUSTIC_FEATURE_DIM, config.ACOUSTIC_FEATURE_DIM * 2),
            nn.ReLU(),
            nn.Linear(config.ACOUSTIC_FEATURE_DIM * 2, llama_hidden_size),
            nn.LayerNorm(llama_hidden_size)
        )
        
        # ✨✨✨ 2. 可学习的温度系数 (Learnable Temperature) ✨✨✨

        # 移动自定义层到设备
        device = torch.device(f"cuda:{local_rank}")
        self.visual_projector.to(device, dtype=torch.bfloat16)
        self.acoustic_projector.to(device, dtype=torch.bfloat16)
        # temperature 自动由 nn.Parameter 管理，DDP 会处理
        self.temperature = nn.Parameter(torch.tensor(0.07)).to(device, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels, visual_features, acoustic_features):
        # 确保特征在正确设备和精度
        device = self.visual_projector[0].weight.device
        visual_features = visual_features.to(device, dtype=torch.bfloat16)
        acoustic_features = acoustic_features.to(device, dtype=torch.bfloat16)

        # ✨✨✨ 3. 投影并进行 Mean Pooling (平均池化) ✨✨✨
        visual_embeds = self.visual_projector(visual_features).mean(dim=1)   # (Batch, Hidden)
        acoustic_embeds = self.acoustic_projector(acoustic_features).mean(dim=1) # (Batch, Hidden)
        
        # 文本嵌入（取<cls> token的嵌入）
        # 这里的 input_ids 已经是包含 prompt+response 的完整序列
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(torch.bfloat16)
        text_cls = text_embeds[:, 0, :] # (Batch, Hidden)
        
        # ========== 对比学习对齐损失 (InfoNCE) ==========
        # 归一化特征
        visual_embeds_norm = F.normalize(visual_embeds, dim=-1)
        acoustic_embeds_norm = F.normalize(acoustic_embeds, dim=-1)
        text_cls_norm = F.normalize(text_cls, dim=-1)
        
        # 计算相似度矩阵
        visual_text_sim = torch.matmul(visual_embeds_norm, text_cls_norm.t()) / self.temperature
        acoustic_text_sim = torch.matmul(acoustic_embeds_norm, text_cls_norm.t()) / self.temperature
        
        # 构建标签（正样本为自身）
        batch_size = input_ids.shape[0]
        # 注意：在 DDP 中，这里的 loss 是 Local Batch 的对比，如果需要 Global 对比需要 gather，
        # 但为了匹配您的代码片段逻辑，这里保持 Local 对比。
        contrast_labels = torch.arange(batch_size, device=input_ids.device)
        
        # 对比损失
        visual_loss = F.cross_entropy(visual_text_sim, contrast_labels)
        acoustic_loss = F.cross_entropy(acoustic_text_sim, contrast_labels)
        contrast_loss = (visual_loss + acoustic_loss) / 2
        
        # ========== 原有生成损失 ==========
        # ✨✨✨ 4. 拼接嵌入 (Visual[1] + Acoustic[1] + Text[Seq]) ✨✨✨
        # 注意: 您的片段是 visual_embeds.unsqueeze(1) (Batch, 1, Hidden)
        inputs_embeds = torch.cat([
            visual_embeds.unsqueeze(1), 
            acoustic_embeds.unsqueeze(1), 
            text_embeds
        ], dim=1)
        
        # 更新 Attention Mask: 前面加了2个模态 Token，所以 Mask 也要加2个 1
        extra_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        final_attn_mask = torch.cat([extra_mask, attention_mask], dim=1)
        
        # 更新 Labels: 前面加了2个模态 Token，Label 设为 -100 (忽略)
        extra_labels = torch.full((labels.shape[0], 2), -100, device=labels.device)
        final_labels = torch.cat([extra_labels, labels], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=final_attn_mask, labels=final_labels, return_dict=True)
        
        # 总损失 = 生成损失 + 0.1 * 对比损失
        total_loss = outputs.loss + 0.5 * contrast_loss
        return total_loss

def create_training_collate_fn(tokenizer, prompt_template):
    def collate_fn(batch):
        # 特征 Padding
        visual_features = pad_sequence([item['visual'] for item in batch], batch_first=True, padding_value=0.0)
        acoustic_features = pad_sequence([item['acoustic'] for item in batch], batch_first=True, padding_value=0.0)

        # 文本构建
        human_prompts_full = []
        assistant_responses = []
        human_template, assistant_template = prompt_template.split("### Assistant:")
        human_template += "### Assistant:"

        for item in batch:
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

        full_texts = [p + r + tokenizer.eos_token for p, r in zip(human_prompts_full, assistant_responses)]
        inputs_tokenized = tokenizer(
            full_texts, padding='longest', return_tensors="pt", truncation=True, max_length=512
        )

        # 标签处理
        labels = inputs_tokenized['input_ids'].clone()
        prompts_tokenized = tokenizer(
            human_prompts_full, padding='longest', return_tensors="pt", truncation=True, max_length=512
        )
        prompt_lengths = torch.sum(prompts_tokenized.attention_mask, dim=1)

        for i in range(len(batch)):
            labels[i, :prompt_lengths[i]] = -100 # Mask Human Prompt
            
            # Mask除数字外的部分 (可选，根据需求)
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
    local_rank, rank, world_size = setup_ddp()
    
    config = Config()
    if is_main_process():
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        print(f"DDP 初始化完成: World Size={world_size}")
    
    if dist.is_initialized(): dist.barrier()
    
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    if is_main_process(): print("\n" + "="*20 + " 2. 初始化模型 " + "="*20)
    
    model = LingoAuraLLM(config, local_rank)
    
    # DDP 包装
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=0.01)

    # ✨✨✨ 修改 3: 分层学习率优化器 ✨✨✨
    print("\n" + "="*20 + " 配置分层学习率 " + "="*20)
    
    # 1. 筛选参数
    # LoRA 参数
    lora_params = [p for n, p in model.named_parameters() if "lora" in n]
    # Projector 参数 (Double MLP)
    projector_params = [p for n, p in model.named_parameters() if "projector" in n]
    
    print(f"LoRA 参数数量: {len(lora_params)}")
    print(f"Projector 参数数量: {len(projector_params)}")

    # 2. 定义优化器
    optimizer = AdamW([
        {'params': lora_params, 'lr': 1e-5},       # LoRA: 小火慢炖
        {'params': projector_params, 'lr': 1e-3}   # Projector: 大火爆炒 (关键！)
    ], weight_decay=0.01)


    if is_main_process(): print("\n" + "="*20 + " 3. 构建数据集 " + "="*20)
    
    train_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_train_fold, 
                                 prompt_template, is_train=True)    
    
    visual_stats = (train_dataset.visual_mean, train_dataset.visual_std)
    acoustic_stats = (train_dataset.acoustic_mean, train_dataset.acoustic_std)

    val_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_valid_fold, 
                               prompt_template, visual_stats=visual_stats, acoustic_stats=acoustic_stats)
    
    collate_fn = create_training_collate_fn(model.module.tokenizer, prompt_template)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              sampler=train_sampler, collate_fn=collate_fn, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            sampler=val_sampler, collate_fn=collate_fn, num_workers=16, pin_memory=True)

    if is_main_process(): print("\n" + "="*20 + " 4. 开始训练 " + "="*20)

    best_val_loss = float('inf') 
    
    for epoch in range(config.EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]") if is_main_process() else train_loader

        optimizer.zero_grad() 
        for i, batch in enumerate(progress_bar):
            device = torch.device(f"cuda:{local_rank}")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss = model(**batch)
            loss.backward()
            
            current_loss = loss.item() 
            train_loss += current_loss
            
            if is_main_process(): progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # 验证循环
        model.eval()
        val_loss_tensor = torch.tensor(0.0).to(f"cuda:{local_rank}")
        count_tensor = torch.tensor(0.0).to(f"cuda:{local_rank}")
        
        with torch.no_grad():
            iterator = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validating]") if is_main_process() else val_loader
            for batch in iterator:
                device = torch.device(f"cuda:{local_rank}")
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch)
                val_loss_tensor += loss
                count_tensor += 1
        
        if dist.is_initialized():
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        
        avg_val_loss = (val_loss_tensor / count_tensor).item()
        
        if is_main_process():
            print(f"Epoch {epoch+1} - Global Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                print(f"发现新的最佳模型！验证损失: {best_val_loss:.4f} -> {avg_val_loss:.4f}。正在保存...")
                best_val_loss = avg_val_loss
                
                save_path = config.OUTPUT_DIR
                os.makedirs(save_path, exist_ok=True)
                raw_model = model.module
                raw_model.llm.save_pretrained(save_path)
                raw_model.tokenizer.save_pretrained(save_path)
                torch.save(raw_model.visual_projector.state_dict(), os.path.join(save_path, "visual_projector.pt"))
                torch.save(raw_model.acoustic_projector.state_dict(), os.path.join(save_path, "acoustic_projector.pt"))
                # 注意：Attention层已被移除，无需保存
                print(f"最佳模型已保存至: {save_path}")
        
        if dist.is_initialized(): dist.barrier()

    if is_main_process():
        stats_to_save = {
            'visual_mean': visual_stats[0].tolist(),
            'visual_std': visual_stats[1].tolist(),
            'acoustic_mean': acoustic_stats[0].tolist(),
            'acoustic_std': acoustic_stats[1].tolist(),
        }
        with open(os.path.join(config.OUTPUT_DIR, 'normalization_stats.json'), 'w') as f:
            json.dump(stats_to_save, f)
        print(f"归一化统计量已保存至: {config.OUTPUT_DIR}")

    cleanup_ddp()

if __name__ == "__main__":
    main()