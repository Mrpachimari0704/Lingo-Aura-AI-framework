# ==============================================================================
#  evaluate.py (Lingo-Aura on CMU-MOSEI - Inference & Evaluation)
# ==============================================================================
#
# HOW TO RUN:
# 1. 确保训练已完成，并且模型权重已保存在 MODEL_PATH 指定的目录中。
# 2. 确保所有依赖项、数据文件和 prompt 模板都与训练时相同。
# 3. 直接运行: python evaluate.py

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


# 导入我们项目中的核心组件
# 假设 evaluate.py 与 lingo_aura_standalone.py 在同一目录
from train_full_model_lora_r16_normalize import  MOSEIDataset
from mmsdk import mmdatasdk as md
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    OUTPUT_DIR = "output/all_model_LoRA_attention_right_label_r16_normalize1"

    LLM_NAME = "./phi-2"
    VISUAL_FEATURE_DIM = 35      # CMU_MOSEI_VisualFacet42 的特征维度
    ACOUSTIC_FEATURE_DIM = 74    # CMU_MOSEI_COVAREP 的特征维度

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    # LEARNING_RATE = 1e-4
    # GRADIENT_A

# --- 1. 推理专用数据处理 ---
# class MOSEIEvaluationDataset(MOSEIDataset):
#     pass
    

# def create_evaluation_collate_fn(tokenizer, prompt_template):
#     def collate_fn(batch):
#         # ✨✨✨【关键修改】: 在这里动态构建推理时需要的 prompt ✨✨✨
        
#         # 从模板中分离出 human 部分
#         human_template, _ = prompt_template.split("### Assistant:")
#         human_template += "### Assistant:"

#         prompts = []
#         for item in batch:
#             # 填充 Human 部分的模板，作为模型的输入
#             prompts.append(
#                 human_template.format(
#                     information_stance=item['cognitive_label'].get("Information Stance", ["N/A"])[0],
#                     reasoning_mode=item['cognitive_label'].get("Reasoning Mode", ["N/A"])[0],
#                     transcription=item['text']
#                 ).strip()
#             )
        
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         # 只对 prompt 部分进行分词
#         tokenized = tokenizer(prompts, padding='longest', return_tensors="pt")

#         return {
#             'input_ids': tokenized['input_ids'],
#             'attention_mask': tokenized['attention_mask'],
#             'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True),
#             'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True),
#             # 'ground_truth_scores' 的来源保持不变
#             'ground_truth_scores': torch.tensor([item['emotion_score'] for item in batch])
#         }
#     return collate_fn


def create_evaluation_collate_fn(tokenizer, prompt_template):
    def collate_fn(batch):
        # --- 🛠️ 强力修正：Few-Shot 引导 ---
        
        # 1. 基础分割
        # 假设 prompt_template 里包含 ### Assistant:
        base_human_part = prompt_template.split("### Assistant:")[0]
        
        # 2. 构建一个“假”的范例 (One-Shot Example)
        # 这是一个教科书级别的示范，告诉模型不要废话，直接给分。
        fake_example = (
            "Information Stance: Neutral. Reasoning Mode: Descriptive. "
            "Transcription: \"The weather is okay, just a normal day.\" "
            "\n### Assistant: Based on the multimodal features, the speaker's emotion score is 0.10."
            "\n### Human: "  # 换行，准备拼接真实的 Prompt
        )
        
        # 3. 真正的强制前缀
        force_prefix = "### Assistant: Based on the multimodal features, the speaker's emotion score is"

        prompts = []
        for item in batch:
            # 填充真实数据的变量
            real_human_text = base_human_part.format(
                information_stance=item['cognitive_label'].get("Information Stance", "N/A"),
                reasoning_mode=item['cognitive_label'].get("Reasoning Mode", "N/A"),
                transcription=item['text']
            )
            
            # 拼接逻辑：[假范例] + [真问题] + [强制前缀]
            full_prompt = fake_example + real_human_text + force_prefix
            
            prompts.append(full_prompt.strip())
        
        # 4. 分词 (注意：推理时 batch_size > 1 必须用 left padding)
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
        self.fake_visual_token = nn.Embedding(1, config.VISUAL_FEATURE_DIM)
        self.fake_acoustic_token = nn.Embedding(1, config.ACOUSTIC_FEATURE_DIM)

        llm_device = 'cuda'
        # 移动嵌入层到GPU
        self.fake_visual_token = self.fake_visual_token.to(llm_device)
        self.fake_acoustic_token = self.fake_acoustic_token.to(llm_device)

    def forward(self, input_ids, attention_mask, visual_features, acoustic_features):
        # 1. 多模态投影与注意力处理（与之前完全一致）
        projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
        projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
        text_embeds = self.base_model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        
        query_embed = text_embeds[:, 0:1, :]
        visual_token_embeds, _ = self.visual_attention(query=query_embed, key=projected_visual, value=projected_visual)
        acoustic_token_embeds, _ = self.acoustic_attention(query=query_embed, key=projected_acoustic, value=projected_acoustic)
        
        inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
        
        extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
        final_attn_mask = torch.cat([attention_mask[:, :1], extra_tokens_mask, attention_mask[:, 1:]], dim=1)

        # 2. 调用生成逻辑 (逻辑不变)
        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attn_mask,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
            min_new_tokens=1
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
    tokenizer.padding_side = "left" 

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

    # ✨✨✨ 2. 加载保存的统计量 ✨✨✨
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
    
    # ✨✨✨ 3. 将统计量传入测试集 Dataset ✨✨✨
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
            # prompt_lengths = [len(ids) for ids in input_ids]
            # generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # ✨✨✨ 添加这行打印语句 ✨✨✨
            print("原始生成内容:", responses) 
            
            # for res in responses:
            #     match = re.search(r"[-+]?\d+(?:\.\d+)?", res)
            #     if match:
            #         pred_score = float(match.group())
            #         all_predictions.append(pred_score)
            #     else:
            #         all_predictions.append(0.0)

            # all_ground_truths.extend(batch['ground_truth_scores'].cpu().numpy())

            for response in responses:
                # 清理空白
                response = response.strip()
                
                # 查找所有数字
                matches = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
                
                if matches:
                    # ✨✨✨ 改为取第一个数字 matches[0] ✨✨✨
                    # 因为我们的 Prompt 结尾是 "score is"，所以紧接着的第一个数字就是分数
                    try:
                        val = float(matches[0])
                        
                        # 范围检查 [-3.5, 3.5] (CMU-MOSEI 范围是 -3 到 3)
                        if -3.5 <= val <= 3.5:
                            all_predictions.append(val)
                        else:
                            # 如果提取出奇怪的数字（比如年份），说明提取错了，由 0.0 兜底
                            all_predictions.append(0.0)
                    except:
                        all_predictions.append(0.0)
                else:
                    # 没找到数字
                    all_predictions.append(0.0)
            
            all_ground_truths.extend(batch['ground_truth_scores'].cpu().numpy())
            print("all_predictions",all_predictions)
            
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



