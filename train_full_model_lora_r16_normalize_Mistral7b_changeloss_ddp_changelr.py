# ==============================================================================
#  train_ddp.py (CMU-MOSEI DDP 分布式训练版)
# ==============================================================================
#
# HOW TO RUN:
# 使用 torchrun 启动，例如使用 4 张显卡：
# torchrun --nproc_per_node=4 train_ddp.py

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0' # DDP 模式下由 torchrun 自动管理，不要硬编码
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist # ✨ DDP 导入
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler # ✨ DDP 导入
from torch.nn.parallel import DistributedDataParallel as DDP # ✨ DDP 导入
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from mmsdk import mmdatasdk as md
from transformers import get_linear_schedule_with_warmup

# --- 0. DDP 初始化辅助函数 ---
def setup_ddp():
    # torchrun 会自动设置这些环境变量
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

# --- 1. 全局配置 ---
class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    OUTPUT_DIR = "output/all_model_LoRA_attention_right_label_r16_normalize_mistral7b_changeloss_ddp_changelr"

    LLM_NAME = "./Mistral-7B-Instruct-v0.2"
    VISUAL_FEATURE_DIM = 35      
    ACOUSTIC_FEATURE_DIM = 74    

    EPOCHS = 20
    # ✨ DDP下，这是单张卡的 Batch Size。总 Batch Size = BATCH_SIZE * GPU数量
    BATCH_SIZE = 10 

# --- 3. 数据加载与处理 (保持不变) ---
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

        # 只在主进程打印，避免刷屏
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
        # mmsdk 在多进程下可能会有些啰嗦，这里不处理，假设内存足够
        temp_dataset = md.mmdataset(recipe)
        features = {
            'visual': temp_dataset[self.visual_field].data,
            'acoustic': temp_dataset[self.acoustic_field].data,
            'labels': temp_dataset[self.label_field].data,
        }
        return features

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
                    aligned[seg_id] = {
                        'visual': vis_feats[vis_mask],
                        'acoustic': acou_feats[acou_mask]
                    }
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

                if np.isnan(visual).any() or np.isinf(visual).any() or \
                np.isnan(acoustic).any() or np.isinf(acoustic).any():
                    continue

                text = self.cognitive_df.loc[segment_id]['text']
                emotion_score = self.cognitive_df.loc[segment_id]['emotion_score']
                cognitive_label = json.loads(self.cognitive_df.loc[segment_id]['cognitive_label'])
                
                prepared_data.append({
                'visual': torch.from_numpy(visual),
                'acoustic': torch.from_numpy(acoustic),
                'text': self.cognitive_df.loc[segment_id]['text'],
                'emotion_score': self.cognitive_df.loc[segment_id]['emotion_score'],
                'cognitive_label': cognitive_label,
                    })
            except Exception: 
                continue
        return prepared_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- 4. 模型架构 ---
class LingoAuraLLM(nn.Module):
    def __init__(self, config, local_rank): # ✨ DDP 修改：传入 local_rank
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # ✨ DDP 关键修改：device_map 必须指定到当前进程的 local_rank
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
        self.visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
        self.acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)
       
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=llama_hidden_size, num_heads=4, batch_first=True
        )
        self.acoustic_attention = nn.MultiheadAttention(
            embed_dim=llama_hidden_size, num_heads=4, batch_first=True
        )

        # ✨ DDP 修改：手动将自定义模块移至 local_rank 设备
        # 注意：不要使用 .to(self.llm.device)，因为 device_map 可能还没生效完全，显式指定更安全
        device = torch.device(f"cuda:{local_rank}")
        self.visual_projector.to(device, dtype=torch.bfloat16)
        self.acoustic_projector.to(device, dtype=torch.bfloat16)
        self.visual_attention.to(device, dtype=torch.bfloat16)
        self.acoustic_attention.to(device, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels, visual_features, acoustic_features):
        projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
        projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        query_embed = text_embeds[:, 0:1, :].to(torch.bfloat16) 

        visual_token_embeds, _ = self.visual_attention(
            query=query_embed, key=projected_visual, value=projected_visual
        )
        # DDP 环境下避免在 forward 里 print，会刷屏
        # print("视觉注意力权重形状:", _ .shape) 

        acoustic_token_embeds, _ = self.acoustic_attention(
            query=query_embed, key=projected_acoustic, value=projected_acoustic
        )

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
        extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        final_attn_mask = torch.cat([attention_mask[:, :1], extra_tokens_mask, attention_mask[:, 1:]], dim=1)
        extra_labels = torch.full((labels.shape[0], 2), -100, device=labels.device)
        final_labels = torch.cat([labels[:, :1], extra_labels, labels[:, 1:]], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=final_attn_mask, labels=final_labels, return_dict=True)
        lm_loss = outputs.loss
        
        text_feat = text_embeds.mean(dim=1).to(torch.bfloat16)
        visual_feat = projected_visual.mean(dim=1).to(torch.bfloat16)
        acoustic_feat = projected_acoustic.mean(dim=1).to(torch.bfloat16)
        contrast_loss = self._contrastive_loss(text_feat, visual_feat, acoustic_feat)
        total_loss = lm_loss + 1 * contrast_loss    ###meanpooling0.5不行，直接上1.0
        return total_loss

    def _contrastive_loss(self, text, visual, acoustic, temperature=0.07):
        text = torch.nn.functional.normalize(text, dim=1)
        visual = torch.nn.functional.normalize(visual, dim=1)
        acoustic = torch.nn.functional.normalize(acoustic, dim=1)
        sim_tv = torch.matmul(text, visual.T) / temperature
        sim_ta = torch.matmul(text, acoustic.T) / temperature
        labels = torch.arange(text.shape[0], device=text.device)
        return (
            torch.nn.functional.cross_entropy(sim_tv, labels) + 
            torch.nn.functional.cross_entropy(sim_ta, labels)
        ) / 2

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

# --- 6. 主执行函数 ---
def main():
    # ✨ 1. DDP 初始化
    local_rank, rank, world_size = setup_ddp()
    
    config = Config()
    
    # 仅 Rank 0 创建目录
    if is_main_process():
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        print("="*20 + " 1. 准备认知标签 " + "="*20)
        print(f"从 '{config.COGNITIVE_LABELS_CSV}' 加载认知标签。")
    
    # 同步，等待主进程创建目录
    dist.barrier()
    
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    if is_main_process():
        print("\n" + "="*20 + " 2. 初始化模型 " + "="*20)
    
    # ✨ 2. 初始化模型 (传入 local_rank)
    model = LingoAuraLLM(config, local_rank)
    
    # ✨ 3. DDP 包装模型
    # find_unused_parameters=False 提高效率，如果报错提示有未使用参数则改为 True
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if is_main_process():
        print("\n" + "="*20 + " 3. 配置优化器 (分层学习率) " + "="*20)
    
    # 注意：DDP 包装后，访问原始模型属性需要通过 model.module
    lora_params = [p for n, p in model.module.named_parameters() if "lora" in n]
    new_module_params = [p for n, p in model.module.named_parameters() if "projector" in n or "attention" in n]
    
    optimizer = AdamW([
        {'params': lora_params, 'lr': 1e-5},       
        {'params': new_module_params, 'lr': 1e-3}  
    ], weight_decay=0.01)

    if is_main_process():
        print("\n" + "="*20 + " 3. 构建数据集 " + "="*20)
        
    train_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_train_fold, 
                                 prompt_template, is_train=True)    
    
    visual_stats = (train_dataset.visual_mean, train_dataset.visual_std)
    acoustic_stats = (train_dataset.acoustic_mean, train_dataset.acoustic_std)

    val_dataset = MOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_valid_fold, 
                               prompt_template, visual_stats=visual_stats, acoustic_stats=acoustic_stats)
    
    # ✨ 4. 配置 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    collate_fn = create_training_collate_fn(model.module.tokenizer, prompt_template)
    
    # ✨ DataLoader: shuffle=False (由 sampler 控制), 传入 sampler
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler, 
                              collate_fn=collate_fn, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, sampler=val_sampler,
                            collate_fn=collate_fn, num_workers=16, pin_memory=True)
    
    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps) 

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    if is_main_process():
        print("\n" + "="*20 + " 4. 开始训练 " + "="*20)

    best_val_loss = float('inf') 
    
    for epoch in range(config.EPOCHS):
        # ✨ 5. 每个 epoch 必须设置 sampler epoch，保证 shuffle 随机性不同
        train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0
        
        # 只在主进程显示进度条
        if is_main_process():
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
        else:
            progress_bar = train_loader
        
        optimizer.zero_grad() 

        for i, batch in enumerate(progress_bar):
            # 移至 local_rank 对应的显卡
            device = torch.device(f"cuda:{local_rank}")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss = model(**batch)
            loss.backward()
            
            current_loss = loss.item() 
            train_loss += current_loss
            
            if is_main_process():
                progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
        
        # 平均训练损失 (仅作参考，这里只算了 rank 0 的，DDP 下通常看验证集即可)
        if is_main_process():
            print(f"Epoch {epoch+1} - Local Train Loss: {train_loss / len(train_loader):.4f}")

        # --- 验证循环 ---
        model.eval()
        val_loss_tensor = torch.tensor(0.0).to(f"cuda:{local_rank}")
        count_tensor = torch.tensor(0.0).to(f"cuda:{local_rank}")
        
        with torch.no_grad():
            iterator = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validating]") if is_main_process() else val_loader
            for batch in iterator:
                device = torch.device(f"cuda:{local_rank}")
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch)
                
                # 累加 loss
                val_loss_tensor += loss
                count_tensor += 1
        
        # ✨ 6. 聚合所有 GPU 的 Loss
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        
        avg_val_loss = (val_loss_tensor / count_tensor).item()
        
        if is_main_process():
            print(f"Epoch {epoch+1} - Global Validation Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                print(f"发现新的最佳模型！验证损失从 {best_val_loss:.4f} 降低到 {avg_val_loss:.4f}。正在保存...")
                best_val_loss = avg_val_loss
                
                save_path = config.OUTPUT_DIR
                os.makedirs(save_path, exist_ok=True)
                
                # ✨ 7. 保存时访问 model.module
                raw_model = model.module
                raw_model.llm.save_pretrained(save_path)
                raw_model.tokenizer.save_pretrained(save_path)
                
                torch.save(raw_model.visual_projector.state_dict(), os.path.join(save_path, "visual_projector.pt"))
                torch.save(raw_model.acoustic_projector.state_dict(), os.path.join(save_path, "acoustic_projector.pt"))
                torch.save(raw_model.visual_attention.state_dict(), os.path.join(save_path, "visual_attention.pt"))
                torch.save(raw_model.acoustic_attention.state_dict(), os.path.join(save_path, "acoustic_attention.pt"))
                
                print(f"最佳模型已保存至: {save_path}")
        
        # 等待保存完成
        dist.barrier()

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
        print("\n" + "="*60)
        print(f"最佳验证损失为: {best_val_loss:.4f}")
        print("="*60)
    
    cleanup_ddp()

if __name__ == "__main__":
    main()