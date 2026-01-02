# ==============================================================================
#  train_ablation_multimodal.py (Ablation Study 2: Multimodal w/o Cognitive Prompt)
# ==============================================================================
#
# HOW TO RUN:
# 1. 确保所有依赖项和数据文件都已准备就绪。
# 2. 确保 prompts/simple_prompt.txt 文件已创建。
# 3. 直接运行: python train_ablation_multimodal.py

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
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

from mmsdk import mmdatasdk as md

# --- 1. 全局配置 ---
class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/simple_prompt.txt" # ✨✨✨ 使用简单模板 ✨✨✨
    OUTPUT_DIR = "output/ablation_multimodal_simple_prompt_attention_epoch15_lr1e-4" # ✨✨✨ 独立的输出目录 ✨✨✨

    LLM_NAME = "./phi-2"
    VISUAL_FEATURE_DIM = 35
    ACOUSTIC_FEATURE_DIM = 74
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 16
    # LEARNING_RATE = 2e-5
    LEARNING_RATE = 1e-4

# --- 2. 基础多模态数据集类 (从你的最终版代码复制而来) ---
# 这个基础类包含了加载和对齐 .csd 文件的核心逻辑
class BaseMOSEIDataset(Dataset):
    def __init__(self, cognitive_df, split_ids, prompt_template):
        self.cognitive_df = cognitive_df.set_index('segment_id')
        self.split_ids = set(split_ids)
        self.prompt_template = prompt_template
        
        self.visual_field = 'CMU_MOSEI_VisualFacet42'
        self.acoustic_field = 'CMU_MOSEI_COVAREP'
        self.label_field = 'CMU_MOSEI_Labels'
        
        print("正在手动加载 .csd 文件...")
        self.features = self._load_csd_files()

        print("正在对齐特征...")
        self.aligned_features = self._align_features()
        
        self.data = self._prepare_data()

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
                    aligned[seg_id] = {
                        'visual': vis_feats[vis_mask],
                        'acoustic': acou_feats[acou_mask]
                    }
        return aligned
    
    def _prepare_data(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- 3. 专用于此消融实验的数据集类 ---
class SimplePromptMOSEIDataset(BaseMOSEIDataset):
    def _prepare_data(self):
        prepared_data = []
        skipped_count = 0
        for segment_id in self.aligned_features.keys():
            if segment_id not in self.split_ids: continue
            try:
                visual_raw = self.aligned_features[segment_id]['visual']
                acoustic_raw = self.aligned_features[segment_id]['acoustic']
                
                if visual_raw.shape[0] == 0 or acoustic_raw.shape[0] == 0:
                    skipped_count += 1
                    continue

                visual = visual_raw.astype(np.float32)
                acoustic = acoustic_raw.astype(np.float32)

                if np.isnan(visual).any() or np.isinf(visual).any() or \
                   np.isnan(acoustic).any() or np.isinf(acoustic).any():
                    skipped_count += 1
                    continue
                
                text = self.cognitive_df.loc[segment_id]['text']
                emotion_score = self.cognitive_df.loc[segment_id]['emotion_score']
                
                # ✨✨✨ 使用简单模板构建 prompt ✨✨✨
                human_prompt = self.prompt_template.split("### Assistant:")[0].format(transcription=text)
                assistant_response = self.prompt_template.split("### Assistant:")[1].format(emotion_score=emotion_score)
                
                prepared_data.append({
                    'visual': torch.from_numpy(visual),
                    'acoustic': torch.from_numpy(acoustic),
                    'full_text': human_prompt + assistant_response,
                    'prompt_len': len(human_prompt)
                })
            except Exception:
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f"警告: 在数据准备过程中，共跳过了 {skipped_count} 个无效样本。")
            
        return prepared_data

# --- 4. 模型架构 (从你的最终版代码复制而来) ---
# class LingoAuraLLM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         quant_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#         )
#         self.llm = AutoModelForCausalLM.from_pretrained(
#             config.LLM_NAME,
#             quantization_config=quant_config,
#             device_map=config.DEVICE,
#             trust_remote_code=True
#         )
#         self.llm = prepare_model_for_kbit_training(self.llm)
        
#         lora_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
#             target_modules=["q_proj", "k_proj", "v_proj", "dense"], 
#             lora_dropout=0.1, bias="none",
#         )
#         self.llm = get_peft_model(self.llm, lora_config)

#         llama_hidden_size = self.llm.config.hidden_size
#         self.visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
#         self.acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)
        
#         llm_device = self.llm.device
#         self.visual_projector.to(llm_device, dtype=torch.bfloat16)
#         self.acoustic_projector.to(llm_device, dtype=torch.bfloat16)

#     def forward(self, input_ids, attention_mask, labels, visual_features, acoustic_features):
#         projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
#         projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
        
#         visual_token_embeds = projected_visual.mean(dim=1, keepdim=True)
#         acoustic_token_embeds = projected_acoustic.mean(dim=1, keepdim=True)
        
#         text_embeds = self.llm.get_input_embeddings()(input_ids)
        
#         inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
        
#         extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
#         final_attn_mask = torch.cat([attention_mask[:, :1], extra_tokens_mask, attention_mask[:, 1:]], dim=1)
        
#         extra_labels = torch.full((labels.shape[0], 2), -100, device=labels.device)
#         final_labels = torch.cat([labels[:, :1], extra_labels, labels[:, 1:]], dim=1)
        
#         outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=final_attn_mask, labels=final_labels, return_dict=True)
#         return outputs.loss


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
            r=8,
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
        
        inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
        extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        final_attn_mask = torch.cat([attention_mask[:, :1], extra_tokens_mask, attention_mask[:, 1:]], dim=1)
        extra_labels = torch.full((labels.shape[0], 2), -100, device=labels.device)
        final_labels = torch.cat([labels[:, :1], extra_labels, labels[:, 1:]], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=final_attn_mask, labels=final_labels, return_dict=True)
        return outputs.loss

# --- 5. 数据整理器 (从你的最终版代码复制而来) ---
def create_collate_fn(tokenizer):
    def collate_fn(batch):
        full_texts = [f['full_text'] for f in batch]
        prompt_lens = [f['prompt_len'] for f in batch]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized = tokenizer(full_texts, padding='longest', truncation=True, max_length=512, return_tensors="pt")
        labels = tokenized['input_ids'].clone()
        for i, p_len_char in enumerate(prompt_lens):
            p_len_token = len(tokenizer.encode(full_texts[i][:p_len_char]))
            labels[i, :p_len_token] = -100
        return {
            'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': labels,
            'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True),
            'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True)
        }
    return collate_fn

# --- 6. 主执行函数 ---
def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*20 + " 1. 准备数据集 " + "="*20)
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    train_dataset = SimplePromptMOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_train_fold, prompt_template)
    val_dataset = SimplePromptMOSEIDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_valid_fold, prompt_template)
    
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
    collate_fn = create_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=16)
    
    print("\n" + "="*20 + " 2. 初始化模型 " + "="*20)
    model = LingoAuraLLM(config)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    
    print("\n" + "="*20 + " 3. 开始训练 " + "="*20)
    best_val_loss = float('inf')
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            loss = model(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - 平均训练损失: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validating]"):
                batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
                loss = model(**batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - 验证损失: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            print(f"发现新的最佳模型！正在保存至: {config.OUTPUT_DIR}")
            best_val_loss = avg_val_loss
            save_path = config.OUTPUT_DIR
            model.llm.save_pretrained(save_path)
            model.tokenizer.save_pretrained(save_path)
            torch.save(model.visual_projector.state_dict(), os.path.join(save_path, "visual_projector.pt"))
            torch.save(model.acoustic_projector.state_dict(), os.path.join(save_path, "acoustic_projector.pt"))

            # ✨✨✨【关键补充】: 保存新的 Attention 模块的权重 ✨✨✨
            print(" - 正在保存注意力模块权重...")
            torch.save(model.visual_attention.state_dict(), os.path.join(save_path, "visual_attention.pt"))
            torch.save(model.acoustic_attention.state_dict(), os.path.join(save_path, "acoustic_attention.pt"))

    print(f"\n训练完成！最佳模型已保存在 {config.OUTPUT_DIR}。")

if __name__ == "__main__":
    main()