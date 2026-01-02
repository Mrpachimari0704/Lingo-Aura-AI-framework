# ==============================================================================
#  evaluate.py (适配 Double MLP + Mean Pooling)
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
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from mmsdk import mmdatasdk as md

# 导入 Dataset (假设在同一目录下)
from train_full_model_lora_r16_normalize import MOSEIDataset

class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    
    # ✨✨✨ 修改 1: 确保路径指向 Double MLP 训练出的模型目录 ✨✨✨
    OUTPUT_DIR = "output/single_card_LoRA_doublemlp_contrastive1.0_warm"

    LLM_NAME = "./Mistral-7B-Instruct-v0.2"
    VISUAL_FEATURE_DIM = 35      
    ACOUSTIC_FEATURE_DIM = 74    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16

def create_evaluation_collate_fn(tokenizer, prompt_template):
    def collate_fn(batch):
        # --- Few-Shot 引导 (保持不变) ---
        base_human_part = prompt_template.split("### Assistant:")[0]
        
        fake_example = (
            "Information Stance: Neutral. Reasoning Mode: Descriptive. "
            "Transcription: \"The weather is okay, just a normal day.\" "
            "\n### Assistant: Based on the multimodal features, the speaker's emotion score is 0.10."
            "\n### Human: " 
        )
        
        force_prefix = "### Assistant: Based on the multimodal features, the speaker's emotion score is"

        prompts = []
        for item in batch:
            real_human_text = base_human_part.format(
                information_stance=item['cognitive_label'].get("Information Stance", "N/A"),
                reasoning_mode=item['cognitive_label'].get("Reasoning Mode", "N/A"),
                transcription=item['text']
            )
            full_prompt = fake_example + real_human_text + force_prefix
            prompts.append(full_prompt.strip())
        
        tokenizer.padding_side = "left" 
        tokenized = tokenizer(prompts, padding='longest', return_tensors="pt")

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True),
            'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True),
            'ground_truth_scores': torch.tensor([item['emotion_score'] for item in batch])
        }
    return collate_fn

# --- ✨✨✨ 修改 2: 推理模型类 (移除 Attention, 加入 Pooling) ✨✨✨ ---
class LingoAuraInferenceModel(nn.Module):
    def __init__(self, config, tokenizer, base_model, visual_projector, acoustic_projector):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.base_model = base_model 
        self.visual_projector = visual_projector
        self.acoustic_projector = acoustic_projector
        # 注意：这里不需要 visual_attention 和 acoustic_attention 了

    def forward(self, input_ids, attention_mask, visual_features, acoustic_features):
        # 1. 投影 (Double MLP)
        projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
        projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
        
        # 2. Mean Pooling (将序列压缩为 1 个 Token)
        # (Batch, Seq, Hidden) -> (Batch, 1, Hidden)
        # 注意：keepdim=True 是为了方便拼接
        visual_embeds = projected_visual.mean(dim=1, keepdim=True)
        acoustic_embeds = projected_acoustic.mean(dim=1, keepdim=True)
        
        # 3. 获取文本 Embedding
        text_embeds = self.base_model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        
        # 4. 拼接: [Visual(1)] + [Acoustic(1)] + [Text(N)]
        # 这里的顺序必须和训练时完全一致！通常 Mean Pooling 放在最前面
        inputs_embeds = torch.cat([visual_embeds, acoustic_embeds, text_embeds], dim=1)

        # 5. 修正 Mask
        # 因为前面加了 2 个模态 Token，Mask 也要在前面补 2 个 1
        extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        final_attn_mask = torch.cat([extra_tokens_mask, attention_mask], dim=1)

        # 6. 生成
        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attn_mask,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False, 
            temperature=0.9, 
            top_p=0.95,        
            top_k=50,  
            min_new_tokens=1
        )

        return outputs

def evaluate():
    print("="*60)
    print("Lingo-Aura LLM - CMU-MOSEI 模型评估脚本 (Double MLP)")
    print("="*60)

    config = Config()
    MODEL_PATH = config.OUTPUT_DIR 
    DEVICE = config.DEVICE

    print(f"\n[1/4] 正在从 '{MODEL_PATH}' 加载模型...")

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
        torch_dtype=torch.bfloat16, 
    )
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.merge_and_unload()
    print(" - LoRA 适配器已加载并融合。")

    llama_hidden_size = model.config.hidden_size

    # ✨✨✨ 修改 3: 重建 Double MLP 结构 (必须与训练代码一致) ✨✨✨
    visual_projector = nn.Sequential(
        nn.Linear(config.VISUAL_FEATURE_DIM, config.VISUAL_FEATURE_DIM * 2),
        nn.ReLU(),
        nn.Linear(config.VISUAL_FEATURE_DIM * 2, llama_hidden_size),
        nn.LayerNorm(llama_hidden_size)
    )
    acoustic_projector = nn.Sequential(
        nn.Linear(config.ACOUSTIC_FEATURE_DIM, config.ACOUSTIC_FEATURE_DIM * 2),
        nn.ReLU(),
        nn.Linear(config.ACOUSTIC_FEATURE_DIM * 2, llama_hidden_size),
        nn.LayerNorm(llama_hidden_size)
    )

    # 加载 Projector 权重
    print(" - 加载 Projector 权重...")
    try:
        visual_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "visual_projector.pt")))
        acoustic_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "acoustic_projector.pt")))
    except FileNotFoundError:
        print("❌ 错误：找不到 Projector 权重文件！请检查 OUTPUT_DIR。")
        return

    # 注意：移除了 Attention 的加载代码

    llm_device = next(model.parameters()).device
    visual_projector.to(device=llm_device, dtype=torch.bfloat16)
    acoustic_projector.to(device=llm_device, dtype=torch.bfloat16)

    # 实例化推理模型 (不再传入 attention 模块)
    model = LingoAuraInferenceModel(
        config=config,
        tokenizer=tokenizer,
        base_model=model,
        visual_projector=visual_projector,
        acoustic_projector=acoustic_projector,
    )
    model.eval() 
    print(" - 多模态推理模型已初始化完成。")

    print(f"\n[2/4] 正在加载 CMU-MOSEI 测试数据集...")

    stats_path = os.path.join(config.OUTPUT_DIR, 'normalization_stats.json')
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        visual_mean = torch.tensor(stats['visual_mean'])
        visual_std = torch.tensor(stats['visual_std'])
        acoustic_mean = torch.tensor(stats['acoustic_mean'])
        acoustic_std = torch.tensor(stats['acoustic_std'])
        
        visual_stats = (visual_mean, visual_std)
        acoustic_stats = (acoustic_mean, acoustic_std)
        print(" - 成功加载归一化统计量。")
    except FileNotFoundError:
        print(f"警告: 找不到归一化文件 {stats_path}。将不进行归一化处理。")
        visual_stats = None
        acoustic_stats = None

    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    test_dataset = MOSEIDataset(
        cognitive_df, 
        md.cmu_mosei.standard_folds.standard_test_fold, 
        prompt_template,
        visual_stats=visual_stats,
        acoustic_stats=acoustic_stats
    )
    collate_fn = create_evaluation_collate_fn(tokenizer, prompt_template)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, collate_fn=collate_fn, num_workers=16)
    
    print(f"\n[3/4] 正在测试集上进行推理...")
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            visual_features = batch['visual_features'].to(config.DEVICE)
            acoustic_features = batch['acoustic_features'].to(config.DEVICE)
            
            # 1. 模型生成
            outputs = model(input_ids, attention_mask, visual_features, acoustic_features)
            
            # 2. 解码
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 3. 提取数字
            if len(all_predictions) == 0:
                print(f"Debug - 纯生成内容: '{responses[0]}'")

            for response in responses:
                response = response.strip()
                matches = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
                
                if matches:
                    # 取第一个数字
                    try:
                        val = float(matches[0])
                        if -3.5 <= val <= 3.5:
                            all_predictions.append(val)
                        else:
                            all_predictions.append(0.0)
                    except:
                        all_predictions.append(0.0)
                else:
                    all_predictions.append(0.0)
            
            all_ground_truths.extend(batch['ground_truth_scores'].cpu().numpy())
            # print("all_predictions", all_predictions) # 可选：打印看进度

    print(f"\n[4/4] 计算评估指标...")
    preds = np.array(all_predictions)
    gts = np.array(all_ground_truths)

    mae = mean_absolute_error(gts, preds)
    acc2 = accuracy_score(gts >= 0, preds >= 0)
    f1 = f1_score(gts >= 0, preds >= 0, average='weighted')
    
    print(f"MAE: {mae:.4f}")
    print(f"Acc-2: {acc2:.4f}")
    print(f"F1: {f1:.4f}")

    if len(preds) > 1 and np.std(preds) > 0:
        print(f"Corr: {np.corrcoef(gts, preds)[0, 1]:.4f}")
    else:
        print("Corr: N/A (方差为0)")

if __name__ == "__main__":
    evaluate()