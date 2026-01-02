# ==============================================================================
#  evaluate.py (Lingo-Aura on CMU-MOSEI - Inference & Evaluation)
# ==============================================================================
#
# HOW TO RUN:
# 1. 确保训练已完成，并且模型权重已保存在 MODEL_PATH 指定的目录中。
# 2. 确保所有依赖项、数据文件和 prompt 模板都与训练时相同。
# 3. 直接运行: python evaluate.py

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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
from train_full_model_lora import  MOSEIDataset
from mmsdk import mmdatasdk as md
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    OUTPUT_DIR = "output/all_model_LoRA_attention_right_label_r16_visionpre_strong_reg"

    LLM_NAME = "./phi-2"
    VISUAL_FEATURE_DIM = 35      # CMU_MOSEI_VisualFacet42 的特征维度
    ACOUSTIC_FEATURE_DIM = 74    # CMU_MOSEI_COVAREP 的特征维度

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    # LEARNING_RATE = 1e-4
    # GRADIENT_A

# --- 1. 推理专用数据处理 ---
class MOSEIEvaluationDataset(MOSEIDataset):
    # __init__, _load_csd_files, _align_features 这些方法都从父类继承，无需重写

    def _prepare_data(self):
        # ✨✨✨【关键修改】: 这个方法现在和新的训练Dataset一样，只打包原始数据 ✨✨✨
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

                # 直接将所有需要的信息打包
                prepared_data.append({
                    'visual': torch.from_numpy(visual),
                    'acoustic': torch.from_numpy(acoustic),
                    'text': self.cognitive_df.loc[segment_id]['text'],
                    'emotion_score': self.cognitive_df.loc[segment_id]['emotion_score'],
                    'cognitive_label': json.loads(self.cognitive_df.loc[segment_id]['cognitive_label']),
                })
            except Exception as e:
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f"警告: 在数据准备过程中，共跳过了 {skipped_count} 个无效样本。")
            
        return prepared_data
    

def create_evaluation_collate_fn(tokenizer, prompt_template):
    def collate_fn(batch):
        # ✨✨✨【关键修改】: 在这里动态构建推理时需要的 prompt ✨✨✨
        
        # 从模板中分离出 human 部分
        human_template, _ = prompt_template.split("### Assistant:")
        human_template += "### Assistant:"

        prompts = []
        for item in batch:
            # 填充 Human 部分的模板，作为模型的输入
            prompts.append(
                human_template.format(
                    information_stance=item['cognitive_label'].get("Information Stance", ["N/A"])[0],
                    reasoning_mode=item['cognitive_label'].get("Reasoning Mode", ["N/A"])[0],
                    transcription=item['text']
                ).strip()
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 只对 prompt 部分进行分词
        tokenized = tokenizer(prompts, padding='longest', return_tensors="pt")

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True),
            'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True),
            # 'ground_truth_scores' 的来源保持不变
            'ground_truth_scores': torch.tensor([item['emotion_score'] for item in batch])
        }
    return collate_fn


class LingoAuraInferenceModel(nn.Module):
    def __init__(self, config, tokenizer, base_model, visual_projector, acoustic_projector, visual_attention, acoustic_attention):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # self.base_model 在这里被赋值。它就是从外部传入的、融合了LoRA的核心LLM。
        self.base_model = base_model 
        
        self.visual_projector = visual_projector
        self.acoustic_projector = acoustic_projector
        self.visual_attention = visual_attention
        self.acoustic_attention = acoustic_attention
        self.hidden_size = base_model.config.hidden_size

    def forward(self, input_ids, attention_mask, visual_features, acoustic_features):
        # 1. 多模态投影与注意力处理（保持不变）
        projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
        projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
        text_embeds = self.base_model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        
        query_embed = text_embeds[:, 0:1, :]
        visual_token_embeds, _ = self.visual_attention(query=query_embed, key=projected_visual, value=projected_visual)
        acoustic_token_embeds, _ = self.acoustic_attention(query=query_embed, key=projected_acoustic, value=projected_acoustic)
        
        # ✨✨✨【核心修改】: 实现与训练时一致的“前缀式融合” ✨✨✨
        
        # A. 拼接输入嵌入 (Input Embeds)
        # 格式: [视觉, 听觉, 原始文本...]
        inputs_embeds = torch.cat([visual_token_embeds, acoustic_token_embeds, text_embeds], dim=1)
        
        # B. 拼接注意力掩码 (Attention Mask)
        # 为2个新的多模态token创建mask
        extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        # 格式: [1, 1, 原始mask...]
        final_attn_mask = torch.cat([extra_tokens_mask, attention_mask], dim=1)

        # 推理时不需要 labels，所以逻辑比训练时更简单

        # 2. 调用生成逻辑
        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attn_mask,
            max_new_tokens=20, # 生成答案的最大长度
            min_new_tokens=1, # ✨ 强制模型至少生成1个token，避免直接输出空
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )
        return outputs
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
    visual_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "visual_projector.pt")))
    acoustic_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "acoustic_projector.pt")))

    # 4. 加载 Attention 模块的权重
    visual_attention.load_state_dict(torch.load(os.path.join(MODEL_PATH, "visual_attention.pt")))
    acoustic_attention.load_state_dict(torch.load(os.path.join(MODEL_PATH, "acoustic_attention.pt")))

    # --- ✨✨✨ 移动设备和类型，并“挂载”到主模型上 ✨✨✨ ---

    llm_device = next(model.parameters()).device
    visual_projector.to(device=llm_device, dtype=torch.bfloat16)
    acoustic_projector.to(device=llm_device, dtype=torch.bfloat16)
    visual_attention.to(device=llm_device, dtype=torch.bfloat16)
    acoustic_attention.to(device=llm_device, dtype=torch.bfloat16)

        # --- ✨ 关键：用自定义模型类整合所有组件 ✨ ---
    model = LingoAuraInferenceModel(
        config=config,
        tokenizer=tokenizer,
        base_model=model,
        visual_projector=visual_projector,
        acoustic_projector=acoustic_projector,
        visual_attention=visual_attention,
        acoustic_attention=acoustic_attention
    )
    model.eval()  # 切换到评估模式
    print(" - 多模态推理模型已初始化完成。")

    print(f"\n[2/4] 正在加载 CMU-MOSEI 测试数据集...")
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    test_dataset = MOSEIEvaluationDataset(cognitive_df, md.cmu_mosei.standard_folds.standard_test_fold, prompt_template)
    collate_fn = create_evaluation_collate_fn(tokenizer, prompt_template)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, collate_fn=collate_fn, num_workers=4)
    
    
    print(f"\n[3/4] 正在测试集上进行推理...")
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):

            # 从batch中提取参数（确保设备一致）
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            visual_features = batch['visual_features'].to(DEVICE)
            acoustic_features = batch['acoustic_features'].to(DEVICE)
            ground_truths = batch['ground_truth_scores']

            # --- ✨ 调用整合后的模型生成结果 ✨ ---
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                acoustic_features=acoustic_features
            )

            # 后续的token解码、分数提取逻辑保持不变
            prompt_lengths = [len(ids) for ids in input_ids]
            generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # ✨✨✨ 添加这行打印语句 ✨✨✨
            print("原始生成内容:", responses) 
            
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
    print("模型的所有预测值:", all_predictions[:50]) # 打印前50个看看
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