# ==============================================================================
#  lingo_aura_standalone_single_dropout.py (单卡版 - 保留Dropout - 无梯度累积)
# ==============================================================================

import os
os.environ['CUDA_VISIBLE_DEVICES']='1' 
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse 
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from mmsdk import mmdatasdk as md
from transformers import get_linear_schedule_with_warmup

# --- 1. 全局配置 ---
class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    # 更新输出目录名以体现单卡和Dropout
    OUTPUT_DIR = "output/single_card_LoRA_doublemlp_dropout_contrastive0.25_warm" 

    LLM_NAME = "./Mistral-7B-Instruct-v0.2"
    VISUAL_FEATURE_DIM = 35
    ACOUSTIC_FEATURE_DIM = 74

    EPOCHS = 20
    
    # [显存设置]
    # 因为去掉了梯度累积，这里的 Batch Size 直接决定更新频率
    # 建议设为 4 或 8。如果显存(VRAM)不足，请改为 2 或 1。
    BATCH_SIZE = 32            
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 数据加载 (保持不变) ---
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

        print("正在加载 .csd 文件...")
        self.features = self._load_csd_files()
        print("正在对齐特征...")
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

# --- 3. 模型架构 (单卡版 - 保留 Dropout) ---
class LingoAuraLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit 量化配置
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # 加载 LLM
        print("正在加载 Mistral-7B (4-bit)...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_NAME,
            quantization_config=quant_config, 
            device_map="auto", 
            trust_remote_code=True,
        )
        
        # 开启梯度检查点 (Gradient Checkpointing) 节省显存
        self.llm.gradient_checkpointing_enable()
        self.llm = prepare_model_for_kbit_training(self.llm)

        # LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  
            lora_dropout=0.05,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        llama_hidden_size = self.llm.config.hidden_size
        
        # ✨✨✨ 1. 双层 MLP 投影 (Double MLP) + Dropout ✨✨✨
        self.visual_projector = nn.Sequential(
            nn.Linear(config.VISUAL_FEATURE_DIM, config.VISUAL_FEATURE_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # ✨ 保留 Dropout
            nn.Linear(config.VISUAL_FEATURE_DIM * 2, llama_hidden_size),
            nn.LayerNorm(llama_hidden_size)
        )
        self.acoustic_projector = nn.Sequential(
            nn.Linear(config.ACOUSTIC_FEATURE_DIM, config.ACOUSTIC_FEATURE_DIM * 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # ✨ 保留 Dropout
            nn.Linear(config.ACOUSTIC_FEATURE_DIM * 2, llama_hidden_size),
            nn.LayerNorm(llama_hidden_size)
        )
        
        # 将自定义层移动到设备
        self.visual_projector.to(config.DEVICE, dtype=torch.bfloat16)
        self.acoustic_projector.to(config.DEVICE, dtype=torch.bfloat16)
        
        # 可学习温度系数
        self.temperature = nn.Parameter(torch.tensor(0.07)).to(config.DEVICE, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels, visual_features, acoustic_features):
        # 确保特征在正确设备和精度
        visual_features = visual_features.to(self.config.DEVICE, dtype=torch.bfloat16)
        acoustic_features = acoustic_features.to(self.config.DEVICE, dtype=torch.bfloat16)

        # 投影并 Mean Pooling
        visual_embeds = self.visual_projector(visual_features).mean(dim=1) 
        acoustic_embeds = self.acoustic_projector(acoustic_features).mean(dim=1)
        
        # 文本嵌入
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(torch.bfloat16)
        text_cls = text_embeds[:, 0, :] 
        
        # ========== 对比学习对齐损失 ==========
        visual_embeds_norm = F.normalize(visual_embeds, p=2, dim=-1, eps=1e-6)
        acoustic_embeds_norm = F.normalize(acoustic_embeds, p=2, dim=-1, eps=1e-6)
        text_cls_norm = F.normalize(text_cls, p=2, dim=-1, eps=1e-6)

        # 限制温度系数
        current_temp = torch.clamp(self.temperature, min=0.01, max=0.5)

        # 计算相似度矩阵
        visual_text_sim = torch.matmul(visual_embeds_norm, text_cls_norm.t()) / current_temp
        acoustic_text_sim = torch.matmul(acoustic_embeds_norm, text_cls_norm.t()) / current_temp

        # 限制 Logits 范围
        visual_text_sim = torch.clamp(visual_text_sim, max=30.0) 
        acoustic_text_sim = torch.clamp(acoustic_text_sim, max=30.0)
        
        batch_size = input_ids.shape[0]
        contrast_labels = torch.arange(batch_size, device=input_ids.device)
        
        visual_loss = F.cross_entropy(visual_text_sim, contrast_labels)
        acoustic_loss = F.cross_entropy(acoustic_text_sim, contrast_labels)
        contrast_loss = (visual_loss + acoustic_loss) / 2
        
        # ========== 生成损失 ==========
        inputs_embeds = torch.cat([
            visual_embeds.unsqueeze(1), 
            acoustic_embeds.unsqueeze(1), 
            text_embeds
        ], dim=1)
        
        extra_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        final_attn_mask = torch.cat([extra_mask, attention_mask], dim=1)
        
        extra_labels = torch.full((labels.shape[0], 2), -100, device=labels.device)
        final_labels = torch.cat([extra_labels, labels], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=final_attn_mask, labels=final_labels, return_dict=True)
        
        # ✨ 保持原本的 0.25 系数
        total_loss = outputs.loss + 0.25 * contrast_loss
        return total_loss

def create_training_collate_fn(tokenizer, prompt_template):
    def collate_fn(batch):
        visual_features = pad_sequence([item['visual'] for item in batch], batch_first=True, padding_value=0.0)
        acoustic_features = pad_sequence([item['acoustic'] for item in batch], batch_first=True, padding_value=0.0)

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

        labels = inputs_tokenized['input_ids'].clone()
        prompts_tokenized = tokenizer(
            human_prompts_full, padding='longest', return_tensors="pt", truncation=True, max_length=512
        )
        prompt_lengths = torch.sum(prompts_tokenized.attention_mask, dim=1)

        for i in range(len(batch)):
            labels[i, :prompt_lengths[i]] = -100 
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

# --- 辅助函数：创建优化器 ---
def create_optimizer_and_scheduler(model, epochs_remaining, steps_per_epoch):
    lora_params = [p for n, p in model.named_parameters() if "lora" in n]
    # Projector 和 Temperature 参数
    projector_params = [p for n, p in model.named_parameters() if "projector" in n or "temperature" in n]
    
    # 筛选
    trainable_lora = [p for p in lora_params if p.requires_grad]
    trainable_projector = [p for p in projector_params if p.requires_grad]
    
    print(f"可训练 LoRA 参数数: {len(trainable_lora)}")
    print(f"可训练 Projector 参数数: {len(trainable_projector)}")

    opt_params = []
    if trainable_lora:
        opt_params.append({'params': trainable_lora, 'lr': 1e-5}) 
    if trainable_projector:
        opt_params.append({'params': trainable_projector, 'lr': 2e-4}) 
        
    optimizer = AdamW(opt_params, weight_decay=0.01)
    
    # 无累积：Total Steps = Epochs * Steps_Per_Epoch
    total_steps = steps_per_epoch * epochs_remaining
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    return optimizer, scheduler

# --- 4. 主执行函数 ---
def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 内存清理
    gc.collect()
    torch.cuda.empty_cache()

    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    print("\n" + "="*20 + " 1. 初始化模型 " + "="*20)
    model = LingoAuraLLM(config)

    print("\n" + "="*20 + " 2. 构建数据集 " + "="*20)
    train_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_train_fold, 
                                 prompt_template, is_train=True)    
    
    visual_stats = (train_dataset.visual_mean, train_dataset.visual_std)
    acoustic_stats = (train_dataset.acoustic_mean, train_dataset.acoustic_std)

    val_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_valid_fold, 
                               prompt_template, visual_stats=visual_stats, acoustic_stats=acoustic_stats)
    
    collate_fn = create_training_collate_fn(model.tokenizer, prompt_template)

    # DataLoader: 单卡直接 Shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_fn, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_fn, num_workers=12, pin_memory=True)

    print("\n" + "="*20 + " 3. 开始训练 " + "="*20)
    best_val_loss = float('inf') 
    WARMUP_EPOCHS = 5     

    print(">>> 初始状态：冻结 LoRA，只训练 Projector (Warmup) <<<")
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = False
        if "projector" in name or "temperature" in name:
            param.requires_grad = True
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, config.EPOCHS, len(train_loader))

    for epoch in range(config.EPOCHS):
        # === 动态解冻逻辑 (Warmup 结束) ===
        if epoch == WARMUP_EPOCHS:
            torch.cuda.empty_cache() 
            print(f"\n{'='*40}")
            print(f"Epoch {epoch+1}: [Warmup 结束] 解冻 LoRA，联合训练...")
            print(f"{'='*40}\n")
            
            # 解冻 LoRA
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            
            # 重建优化器 (加入新解冻的参数)
            remaining_epochs = config.EPOCHS - epoch
            optimizer, scheduler = create_optimizer_and_scheduler(model, remaining_epochs, len(train_loader))

        # === 训练循环 (无梯度累积版) ===
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

        for batch in progress_bar:
            # 1. 梯度清零
            optimizer.zero_grad() 
            
            # 2. 数据上卡
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            # 3. 前向计算
            loss = model(**batch)
            
            # 4. 反向传播 (NaN 检查)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Loss is NaN/Inf, skipping step.")
                continue # 直接跳过该 Batch

            loss.backward()
            
            # 5. 梯度裁剪与更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # === 验证循环 ===
        model.eval()
        val_loss = 0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]"):
                batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
                loss = model(**batch)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    count += 1
        
        avg_val_loss = val_loss / count if count > 0 else float('inf')
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        
        # === 保存最佳模型 ===
        if avg_val_loss < best_val_loss:
            print(f"New Best Model! Saving... ({best_val_loss:.4f} -> {avg_val_loss:.4f})")
            best_val_loss = avg_val_loss
            
            save_path = config.OUTPUT_DIR
            model.llm.save_pretrained(save_path) # 保存 LoRA
            model.tokenizer.save_pretrained(save_path)
            torch.save(model.visual_projector.state_dict(), os.path.join(save_path, "visual_projector.pt"))
            torch.save(model.acoustic_projector.state_dict(), os.path.join(save_path, "acoustic_projector.pt"))
            print(f"Saved to: {save_path}")

    # 保存统计数据
    stats_to_save = {
        'visual_mean': visual_stats[0].tolist(),
        'visual_std': visual_stats[1].tolist(),
        'acoustic_mean': acoustic_stats[0].tolist(),
        'acoustic_std': acoustic_stats[1].tolist(),
    }
    with open(os.path.join(config.OUTPUT_DIR, 'normalization_stats.json'), 'w') as f:
        json.dump(stats_to_save, f)

if __name__ == "__main__":
    main()