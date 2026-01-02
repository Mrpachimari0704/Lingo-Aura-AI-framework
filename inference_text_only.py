# ==============================================================================
#  evaluate_baseline_text_only.py (Evaluation for Text-Only Baseline)
# ==============================================================================
#  此脚本为完整独立版，可直接复制运行
# ==============================================================================

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from mmsdk import mmdatasdk as md
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 1. 配置 ---
class Config:
    MODEL_PATH = "output/ablation_text_only"
    PROMPT_TEMPLATE_PATH = "./prompts/simple_prompt.txt"
    COGNITIVE_LABELS_CSV = "./data/cmumosei/cmu_mosei_with_cognitive_labels_v4.csv"
    LLM_NAME = "./phi-2"
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 推理专用数据处理 (纯文本版) ---
class TextOnlyEvaluationDataset(Dataset):
    def __init__(self, cognitive_df, split_ids, prompt_template):
        self.cognitive_df = cognitive_df.set_index('segment_id')
        self.split_ids = set(split_ids)
        self.prompt_template = prompt_template
        self.data = self._prepare_data()

    def _prepare_data(self):
        prepared_data = []
        for segment_id in self.split_ids:
            try:
                if segment_id not in self.cognitive_df.index: continue
                text = self.cognitive_df.loc[segment_id]['text']
                emotion_score = self.cognitive_df.loc[segment_id]['emotion_score']
                human_prompt = self.prompt_template.split("### Assistant:")[0].format(transcription=text)
                prepared_data.append({
                    'prompt': human_prompt + "### Assistant:",
                    'ground_truth_score': emotion_score
                })
            except Exception:
                continue
        return prepared_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def create_text_only_collate_fn(tokenizer):
    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenized = tokenizer(prompts, padding='longest', return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'ground_truth_scores': torch.tensor([item['ground_truth_score'] for item in batch])
        }
    return collate_fn

# --- 3. 主评估函数 ---
def evaluate_text_only():
    config = Config()
    print("="*60)
    print("Lingo-Aura - 纯文本基线模型评估脚本")
    print(f"评估模型路径: {config.MODEL_PATH}")
    print("="*60)

    print(f"\n[1/4] 正在从 '{config.MODEL_PATH}' 加载模型...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(config.LLM_NAME, quantization_config=quant_config, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, config.MODEL_PATH)
    model = model.merge_and_unload()
    print(" - LoRA 适配器已加载并融合。")
    model.eval()

    print(f"\n[2/4] 正在加载 CMU-MOSEI 测试数据集...")
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    test_dataset = TextOnlyEvaluationDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_test_fold, prompt_template)
    collate_fn = create_text_only_collate_fn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=4)

    print(f"\n[3/4] 正在测试集上进行推理...")
    all_predictions, all_ground_truths = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Text-Only Model"):
            batch_on_device = {k: v.to(config.DEVICE) for k, v in batch.items() if hasattr(v, 'to')}
            
            outputs = model.generate(
                input_ids=batch_on_device['input_ids'],
                attention_mask=batch_on_device['attention_mask'],
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_lengths = [len(ids) for ids in batch['input_ids']]
            generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print("原始生成内容:", responses) 


            for res in responses:
                match = re.search(r"[-+]?\d+(?:\.\d+)?", res)
                if match: all_predictions.append(float(match.group()))
                else: all_predictions.append(0.0)

            all_ground_truths.extend(batch['ground_truth_scores'].cpu().numpy())

    print(f"\n[4/4] 计算评估指标...")
    print("\n" + "="*20 + " 纯文本基线评估结果 " + "="*20)
    gts = np.array(all_ground_truths)
    preds = np.array(all_predictions)
    mae = np.mean(np.abs(gts - preds))
    print(f"平均绝对误差 (MAE) ↓: {mae:.4f}")
    acc2 = accuracy_score(gts >= 0, preds >= 0)
    print(f"二元准确率 (Acc-2) ↑: {acc2*100:.2f}%")
    f1 = f1_score(gts >= 0, preds >= 0, average='weighted')
    print(f"加权F1分数 (F1-Score) ↑: {f1:.4f}")
    valid_indices = ~np.isnan(gts) & ~np.isnan(preds)
    if np.sum(valid_indices) > 1:
        corr = np.corrcoef(gts[valid_indices], preds[valid_indices])[0, 1]
        print(f"皮尔逊相关系数 (Corr) ↑: {corr:.4f}")
    else:
        print("无法计算相关系数 (有效数据点不足)。")
    print("="*54)

if __name__ == "__main__":
    evaluate_text_only()