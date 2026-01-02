# ==============================================================================
#  evaluate.py (Lingo-Aura on CMU-MOSEI - Inference & Evaluation)
# ==============================================================================
#
# HOW TO RUN:
# 1. 确保训练已完成，并且模型权重已保存在 MODEL_PATH 指定的目录中。
# 2. 确保所有依赖项、数据文件和 prompt 模板都与训练时相同。
# 3. 直接运行: python evaluate.py

import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


# 导入我们项目中的核心组件
# 假设 evaluate.py 与 lingo_aura_standalone.py 在同一目录
from train_full_model import Config, MOSEIDataset
from mmsdk import mmdatasdk as md
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# --- 1. 推理专用数据处理 ---
class MOSEIEvaluationDataset(MOSEIDataset):
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
                cognitive_label = json.loads(self.cognitive_df.loc[segment_id]['cognitive_label'])
                
                human_prompt = self.prompt_template.split("### Assistant:")[0].format(
                    information_stance=cognitive_label.get("Information Stance", "N/A"),
                    reasoning_mode=cognitive_label.get("Reasoning Mode", "N/A"),
                    transcription=text
                )
                
                prepared_data.append({
                    'visual': torch.from_numpy(visual),
                    'acoustic': torch.from_numpy(acoustic),
                    'prompt': human_prompt + "### Assistant:",
                    'ground_truth_score': emotion_score
                })
            except Exception as e:
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f"警告: 在数据准备过程中，共跳过了 {skipped_count} 个无效样本。")
            
        return prepared_data

def create_evaluation_collate_fn(tokenizer):
    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenized = tokenizer(prompts, padding='longest', return_tensors="pt")

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True),
            'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True),
            'ground_truth_scores': torch.tensor([item['ground_truth_score'] for item in batch])
        }
    return collate_fn

# --- 主评估函数 ---
def evaluate():
    print("="*60)
    print("Lingo-Aura LLM - CMU-MOSEI 模型评估脚本")
    print("="*60)

    config = Config()
    MODEL_PATH = config.OUTPUT_DIR 
    DEVICE = config.DEVICE

    # --- [1/4] 模型加载部分（已修改） ---
    print(f"\n[1/4] 正在从 '{MODEL_PATH}' 加载模型...")

    # a. 在加载基础模型时，就指定好最终的 torch_dtype
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.LLM_NAME,
        quantization_config=quant_config,
        device_map=config.DEVICE,
        trust_remote_code=True,
        # ✨✨✨【关键修改 1】✨✨✨
        # 在加载时就明确指定计算和权重的dtype，防止后续转换
        torch_dtype=torch.bfloat16, 
    )
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # b. 将LoRA适配器融合到基础模型中
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.merge_and_unload()
    print(" - LoRA 适配器已加载并融合。")

    # c. 创建并加载 Projectors
    llama_hidden_size = model.config.hidden_size
    visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
    acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)

    visual_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "visual_projector.pt")))
    acoustic_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "acoustic_projector.pt")))

    # ✨✨✨【关键修改 2】✨✨✨
    # 手动将Projectors移动到与LLM相同的设备和dtype上
    # LLM已经被device_map分配好了，我们让projectors跟随它
    llm_device = next(model.parameters()).device
    visual_projector.to(device=llm_device, dtype=torch.bfloat16)
    acoustic_projector.to(device=llm_device, dtype=torch.bfloat16)

    # 将Projectors作为属性附加到模型上
    model.visual_projector = visual_projector
    model.acoustic_projector = acoustic_projector

    print(" - 多模态 Projectors 已加载。")
    model.eval()

    print(f"\n[2/4] 正在加载 CMU-MOSEI 测试数据集...")
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    test_dataset = MOSEIEvaluationDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_test_fold, prompt_template)
    collate_fn = create_evaluation_collate_fn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, collate_fn=collate_fn, num_workers=4)

    print(f"\n[3/4] 正在测试集上进行推理...")
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # 准备多模态嵌入
            visual_features = batch['visual_features'].to(torch.bfloat16)
            acoustic_features = batch['acoustic_features'].to(torch.bfloat16)
            
            projected_visual = model.visual_projector(visual_features)
            projected_acoustic = model.acoustic_projector(acoustic_features)
            visual_token_embeds = projected_visual.mean(dim=1, keepdim=True)
            acoustic_token_embeds = projected_acoustic.mean(dim=1, keepdim=True)
            
            text_embeds = model.get_input_embeddings()(batch['input_ids'])
            inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
            
            # ✨✨✨【关键修改 2】: 构建并传入与 inputs_embeds 匹配的 attention_mask ✨✨✨
            # 原始文本的 attention_mask
            text_attention_mask = batch['attention_mask']
            # 为新插入的2个多模态token创建mask (batch_size, 2)
            extra_tokens_mask = torch.ones((text_attention_mask.shape[0], 2), device=DEVICE, dtype=torch.long)
            # 拼接成最终的 attention_mask
            attention_mask = torch.cat([text_attention_mask[:, :1], extra_tokens_mask, text_attention_mask[:, 1:]], dim=1)

            # 模型生成
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, # 传入 attention_mask
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True, temperature=0.7, top_p=0.9
            )

            # prompt_lengths = [len(ids) for ids in batch['input_ids']]
            # generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print("outputs：",responses)

            for res in responses:
                match = re.search(r"[-+]?\d+(?:\.\d+)?", res)
                if match:
                    pred_score = float(match.group())
                    all_predictions.append(pred_score)
                else:
                    all_predictions.append(0.0)

            all_ground_truths.extend(batch['ground_truth_scores'].cpu().numpy())

    print(f"\n[4/4] 计算评估指标...")
    print("\n" + "="*20 + " 最终评估结果 " + "="*20)
    gts = np.array(all_ground_truths)
    preds = np.array(all_predictions)

    mae = np.mean(np.abs(gts - preds))
    print(f"平均绝对误差 (MAE) ↓: {mae:.4f}")

    acc2 = accuracy_score(gts >= 0, preds >= 0)
    print(f"二元准确率 (Acc-2) ↑: {acc2*100:.2f}%")

    f1 = f1_score(gts >= 0, preds >= 0, average='weighted')
    print(f"加权F1分数 (F1-Score) ↑: {f1:.4f}")

    # 修改后
    valid_indices = ~np.isnan(gts) & ~np.isnan(preds)
    gts_valid = gts[valid_indices]
    preds_valid = preds[valid_indices]

    # ✨✨✨ 核心修复：检查标准差是否为0 ✨✨✨
    if gts_valid.size > 1 and preds_valid.size > 1 and \
    np.std(gts_valid) > 0 and np.std(preds_valid) > 0:
        
        corr = np.corrcoef(gts_valid, preds_valid)[0, 1]
        print(f"皮尔逊相关系数 (Corr) ↑: {corr:.4f}")
    else:
        print("无法计算相关系数 (原因: 预测值或真实值的标准差为0，或有效数据点不足)。")


if __name__ == "__main__":
    evaluate()