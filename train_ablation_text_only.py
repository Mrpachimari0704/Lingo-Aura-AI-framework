# ==============================================================================
#  train_baseline_text_only.py (Ablation Study 1: Text-Only)
# ==============================================================================
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
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
    PROMPT_TEMPLATE_PATH = "./prompts/simple_prompt.txt" # 使用简单模板
    OUTPUT_DIR = "output/ablation_text_only" # ✨✨✨ 独立的输出目录 ✨✨✨

    LLM_NAME = "./phi-2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5

# --- 2. 数据加载 (仅文本) ---
class TextOnlyMOSEIDataset(Dataset):
    def __init__(self, cognitive_df, split_ids, prompt_template):
        self.cognitive_df = cognitive_df.set_index('segment_id')
        self.split_ids = set(split_ids)
        self.prompt_template = prompt_template
        self.data = self._prepare_data()

    def _prepare_data(self):
        prepared_data = []
        for segment_id in self.split_ids:
            try:
                # 只需确保 segment_id 在 cognitive_df 中存在
                if segment_id not in self.cognitive_df.index:
                    continue
                
                text = self.cognitive_df.loc[segment_id]['text']
                emotion_score = self.cognitive_df.loc[segment_id]['emotion_score']
                
                # ✨✨✨ 只使用文本构建 prompt ✨✨✨
                human_prompt = self.prompt_template.split("### Assistant:")[0].format(transcription=text)
                assistant_response = self.prompt_template.split("### Assistant:")[1].format(emotion_score=emotion_score)
                
                prepared_data.append({
                    'full_text': human_prompt + assistant_response,
                    'prompt_len': len(human_prompt)
                })
            except Exception:
                continue
        return prepared_data
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- 3. 模型架构 (仅文本) ---
# ✨✨✨ 模型简化为纯文本LLM，不需要LingoAuraLLM类 ✨✨✨

# --- 4. 数据整理器 (仅文本) ---
def create_text_only_collate_fn(tokenizer):
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
            
        # ✨✨✨ 只返回文本相关数据 ✨✨✨
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': labels}
    return collate_fn

# --- 5. 主执行函数 ---
def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # --- 数据加载 ---
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    train_ids = md.cmu_mosei.standard_folds.standard_train_fold
    val_ids = md.cmu_mosei.standard_folds.standard_valid_fold

    train_dataset = TextOnlyMOSEIDataset(cognitive_df, train_ids, prompt_template)
    val_dataset = TextOnlyMOSEIDataset(cognitive_df, val_ids, prompt_template)
    
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
    collate_fn = create_text_only_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=16)
    
    # --- 模型初始化 (简化) ---
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(config.LLM_NAME, quantization_config=quant_config, device_map=config.DEVICE)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "dense"], lora_dropout=0.1, bias="none")
    model = get_peft_model(model, lora_config)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    
    # --- 训练与验证循环 ---
    best_val_loss = float('inf')
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
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
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - 验证损失: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            print(f"发现新的最佳模型！正在保存至: {config.OUTPUT_DIR}")
            best_val_loss = avg_val_loss
            model.save_pretrained(config.OUTPUT_DIR)
            tokenizer.save_pretrained(config.OUTPUT_DIR)

    print(f"\n训练完成！最佳模型已保存在 {config.OUTPUT_DIR}。")

if __name__ == "__main__":
    main()