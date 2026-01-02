# ==============================================================================
#  evaluate_ablation_multimodal.py (Evaluation for Multimodal w/o Cognitive Prompt)
# ==============================================================================
#  此脚本为完整独立版，可直接复制运行
# ==============================================================================

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
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

from mmsdk import mmdatasdk as md
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
# --- 1. 配置 ---
class Config:
    MODEL_PATH = "output/ablation_multimodal_simple_prompt_attention"
    PROMPT_TEMPLATE_PATH = "./prompts/simple_prompt.txt"
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    LLM_NAME = "./phi-2"
    VISUAL_FEATURE_DIM = 35
    ACOUSTIC_FEATURE_DIM = 74
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 基础多模态数据集类 (包含加载和对齐逻辑) ---
class BaseMOSEIDataset(Dataset):
    def __init__(self, cognitive_df, split_ids, prompt_template):
        self.config = Config()
        self.cognitive_df = cognitive_df.set_index('segment_id')
        self.split_ids = set(split_ids)
        self.prompt_template = prompt_template
        self.visual_field = 'CMU_MOSEI_VisualFacet42'
        self.acoustic_field = 'CMU_MOSEI_COVAREP'
        self.label_field = 'CMU_MOSEI_Labels'
        self.features = self._load_csd_files()
        self.aligned_features = self._align_features()
        self.data = self._prepare_data()

    def _load_csd_files(self):
        recipe = {
            self.visual_field: os.path.join(self.config.DATA_PATH, self.visual_field + '.csd'),
            self.acoustic_field: os.path.join(self.config.DATA_PATH, self.acoustic_field + '.csd'),
            self.label_field: os.path.join(self.config.DATA_PATH, self.label_field + '.csd'),
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
    
    def _prepare_data(self): raise NotImplementedError
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- 3. 推理专用数据处理 (继承自Base) ---
class SimplePromptEvaluationDataset(BaseMOSEIDataset):
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
                human_prompt = self.prompt_template.split("### Assistant:")[0].format(transcription=text)
                prepared_data.append({
                    'visual': torch.from_numpy(visual),
                    'acoustic': torch.from_numpy(acoustic),
                    'prompt': human_prompt + "### Assistant:",
                    'ground_truth_score': emotion_score
                })
            except Exception as e:
                skipped_count += 1
                continue
        if skipped_count > 0: print(f"警告: 跳过了 {skipped_count} 个无效样本。")
        return prepared_data

def create_evaluation_collate_fn(tokenizer):
    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        tokenized = tokenizer(prompts, padding='longest', return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True),
            'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True),
            'ground_truth_scores': torch.tensor([item['ground_truth_score'] for item in batch])
        }
    return collate_fn

# --- 4. 主评估函数 ---
def evaluate_multimodal_simple():
    config = Config()
    print("="*60)
    print("Lingo-Aura - 多模态（无认知提示）模型评估脚本")
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
    llama_hidden_size = model.config.hidden_size

    #开始重建架构 ✨✨✨ ---
    # 1. 创建 Projector 模块的“空壳”
    visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
    acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)

    # 2. 创建 Attention 模块的“空壳”
    #    这里的参数（embed_dim, num_heads）必须与训练时完全一致！
    visual_attention = nn.MultiheadAttention(embed_dim=llama_hidden_size, num_heads=4, batch_first=True)
    acoustic_attention = nn.MultiheadAttention(embed_dim=llama_hidden_size, num_heads=4, batch_first=True)

    # --- ✨✨✨ 加载权重 ✨✨✨ ---

    # 3. 加载 Projector 的权重
    visual_projector.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, "visual_projector.pt")))
    acoustic_projector.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, "acoustic_projector.pt")))

    # 4. 加载 Attention 模块的权重
    visual_attention.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, "visual_attention.pt")))
    acoustic_attention.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, "acoustic_attention.pt")))

    # --- ✨✨✨ 移动设备和类型，并“挂载”到主模型上 ✨✨✨ ---

    llm_device = next(model.parameters()).device
    visual_projector.to(device=llm_device, dtype=torch.bfloat16)
    acoustic_projector.to(device=llm_device, dtype=torch.bfloat16)
    visual_attention.to(device=llm_device, dtype=torch.bfloat16)
    acoustic_attention.to(device=llm_device, dtype=torch.bfloat16)

    # 5. 将这些加载好权重的模块，作为属性“挂载”到已经融合了LoRA的LLM模型上
    model.visual_projector = visual_projector
    model.acoustic_projector = acoustic_projector
    model.visual_attention = visual_attention
    model.acoustic_attention = acoustic_attention

    print(" - 多模态 Projectors 及 Attention 模块已加载。")
    model.eval()

    print(f"\n[2/4] 正在加载 CMU-MOSEI 测试数据集...")
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    test_dataset = SimplePromptEvaluationDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_test_fold, prompt_template)
    collate_fn = create_evaluation_collate_fn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=4)

    print(f"\n[3/4] 正在测试集上进行推理...")
    all_predictions, all_ground_truths = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Multimodal (Simple Prompt)"):
            batch_on_device = {k: v.to(config.DEVICE) for k, v in batch.items() if hasattr(v, 'to')}
            visual_features = batch_on_device['visual_features'].to(torch.bfloat16)
            acoustic_features = batch_on_device['acoustic_features'].to(torch.bfloat16)
            projected_visual = model.visual_projector(visual_features)
            projected_acoustic = model.acoustic_projector(acoustic_features)
            
            ###注意力
            text_embeds = model.get_input_embeddings()(batch['input_ids']).to(torch.bfloat16) # 需要先获取text_embeds
            query_embed = text_embeds[:, 0:1, :]

            visual_token_embeds, _ = model.visual_attention(query=query_embed, key=projected_visual, value=projected_visual)
            acoustic_token_embeds, _ = model.acoustic_attention(query=query_embed, key=projected_acoustic, value=projected_acoustic)
            
            
            text_embeds = model.get_input_embeddings()(batch_on_device['input_ids'])
            inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
            text_attention_mask = batch_on_device['attention_mask']
            extra_tokens_mask = torch.ones((text_attention_mask.shape[0], 2), device=config.DEVICE, dtype=torch.long)
            attention_mask = torch.cat([text_attention_mask[:, :1], extra_tokens_mask, text_attention_mask[:, 1:]], dim=1)
            outputs = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
            prompt_lengths = [len(ids) for ids in batch['input_ids']]
            generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for res in responses:
                match = re.search(r"[-+]?\d+(?:\.\d+)?", res)
                if match: all_predictions.append(float(match.group()))
                else: all_predictions.append(0.0)
            all_ground_truths.extend(batch['ground_truth_scores'].cpu().numpy())

    print(f"\n[4/4] 计算评估指标...")
    print("\n" + "="*20 + " 多模态（无认知提示）评估结果 " + "="*20)
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
    evaluate_multimodal_simple()