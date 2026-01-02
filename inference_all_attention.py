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
from train_full_model_lora import  MOSEIDataset
from mmsdk import mmdatasdk as md
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    OUTPUT_DIR = "output/all_model_LoRA_attention"

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





# class LingoAuraInferenceModel(nn.Module):
#     def __init__(self, config, tokenizer, base_model, visual_projector, acoustic_projector, visual_attention, acoustic_attention):
#         super().__init__()
#         self.config = config
#         self.tokenizer = tokenizer
#         self.base_model = base_model
#         self.visual_projector = visual_projector
#         self.acoustic_projector = acoustic_projector
#         self.visual_attention = visual_attention
#         self.acoustic_attention = acoustic_attention
#         self.hidden_size = base_model.config.hidden_size

#     def forward(self, input_ids, attention_mask, visual_features, acoustic_features):
#         # 多模态投影与注意力处理（与训练时逻辑完全一致）
#         projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
#         projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
#         text_embeds = self.base_model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        
#         query_embed = text_embeds[:, 0:1, :]
#         visual_token_embeds, _ = self.visual_attention(query=query_embed, key=projected_visual, value=projected_visual)
#         acoustic_token_embeds, _ = self.acoustic_attention(query=query_embed, key=projected_acoustic, value=projected_acoustic)
        
#         inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
        
#         inputs_text_part = torch.cat([
#             inputs_embeds[:, :1, :],        # 对应原文本第0个token
#             inputs_embeds[:, 3:, :]         # 对应原文本第1个token及以后
#         ], dim=1)  # 形状: (batch, L, hidden)，与text_embeds一致


#         # 检查新生成的Token是否接近于0
#         visual_token_norm = torch.mean(torch.abs(visual_token_embeds)).item()
#         acoustic_token_norm = torch.mean(torch.abs(acoustic_token_embeds)).item()

#         print(f"平均视觉Token范数: {visual_token_norm:.6f}")
#         print(f"平均声学Token范数: {acoustic_token_norm:.6f}")

#         if visual_token_norm < 1e-5 or acoustic_token_norm < 1e-5:
#             print("警告：新生成的多模态Token接近于零，模块可能未工作！")

            
#         # 计算重叠部分的差异
#         embed_diff = torch.mean(torch.abs(inputs_text_part - text_embeds)).item()
#         print(f"多模态嵌入中，文本部分与纯文本嵌入的平均差异: {embed_diff:.6f}")
#         if embed_diff < 1e-5:
#             print("警告：多模态特征未改变文本嵌入，模块可能未工作！")
        

#         extra_tokens_mask = torch.ones((attention_mask.shape[0], 2), device=attention_mask.device)
#         final_attn_mask = torch.cat([attention_mask[:, :1], extra_tokens_mask, attention_mask[:, 1:]], dim=1)
        
#         # 调用base_model的生成逻辑
#         outputs = self.base_model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=final_attn_mask,
#             max_new_tokens=20,
#             pad_token_id=self.tokenizer.eos_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#             do_sample=False,
#             temperature=0.0
#         )
#         return outputs




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

        # # ==================== ✨✨✨ 诊断代码块 (已修复) ✨✨✨ ====================
        # if not self.training:
        #     # [步骤 A] 纯文本前向传播，✨✨✨ 关键修复：添加 output_hidden_states=True ✨✨✨
        #     outputs_text_only = self.base_model(
        #         inputs_embeds=text_embeds,
        #         attention_mask=attention_mask,
        #         output_hidden_states=True,  # 强制模型输出隐藏状态
        #     )
        #     # ✨✨✨ 关键修复：从 hidden_states 元组中获取最后一层 ✨✨✨
        #     final_text_hidden_state_text_only = outputs_text_only.hidden_states[-1]

        #     # [步骤 B] 多模态前向传播，✨✨✨ 关键修复：添加 output_hidden_states=True ✨✨✨
        #     outputs_multimodal = self.base_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=final_attn_mask,
        #         output_hidden_states=True,  # 强制模型输出隐藏状态
        #     )
        #     # ✨✨✨ 关键修复：从 hidden_states 元组中获取最后一层 ✨✨✨
        #     last_hidden_state_mm = outputs_multimodal.hidden_states[-1]

        #     # [步骤 C] 从多模态输出中提取文本部分 (逻辑不变)
        #     final_text_hidden_state_mm = torch.cat([
        #         last_hidden_state_mm[:, :1, :], 
        #         last_hidden_state_mm[:, 3:, :]
        #     ], dim=1)

        #     # [步骤 D] 比较差异并打印 (逻辑不变)
        #     min_len = min(final_text_hidden_state_mm.shape[1], final_text_hidden_state_text_only.shape[1])
        #     diff = torch.abs(
        #         final_text_hidden_state_mm[:, :min_len, :] - final_text_hidden_state_text_only[:, :min_len, :]
        #     )
        #     final_diff = torch.mean(diff).item()

        #     print(f"最终隐藏层诊断：文本表示与纯文本模式的平均差异: {final_diff:.6f}")

        #     if final_diff < 1e-4:
        #         print("警告[方法二]：多模态特征未显著改变文本的最终表示！模块可能未被有效利用。")
        # # ==================== ✨✨✨ 诊断代码块结束 ✨✨✨ ====================
        
        # 2. 调用生成逻辑 (逻辑不变)
        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attn_mask,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0
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

    # # 5. 将这些加载好权重的模块，作为属性“挂载”到已经融合了LoRA的LLM模型上
    # model.visual_projector = visual_projector
    # model.acoustic_projector = acoustic_projector
    # model.visual_attention = visual_attention
    # model.acoustic_attention = acoustic_attention

    # print(" - 多模态 Projectors 及 Attention 模块已加载。")
    # model.eval()

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
    collate_fn = create_evaluation_collate_fn(tokenizer)

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, collate_fn=collate_fn, num_workers=4)

    print(f"\n[3/4] 正在测试集上进行推理...")
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            # batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # # 准备多模态嵌入
            # visual_features = batch['visual_features'].to(torch.bfloat16)
            # acoustic_features = batch['acoustic_features'].to(torch.bfloat16)
            
            # projected_visual = model.visual_projector(visual_features)
            # projected_acoustic = model.acoustic_projector(acoustic_features)

            # ##注意力
            # text_embeds = model.get_input_embeddings()(batch['input_ids']).to(torch.bfloat16) # 需要先获取text_embeds
            # query_embed = text_embeds[:, 0:1, :]

            # visual_token_embeds, _ = model.visual_attention(query=query_embed, key=projected_visual, value=projected_visual)
            # acoustic_token_embeds, _ = model.acoustic_attention(query=query_embed, key=projected_acoustic, value=projected_acoustic)
                        
            # text_embeds = model.get_input_embeddings()(batch['input_ids'])
            # inputs_embeds = torch.cat([text_embeds[:, :1, :], visual_token_embeds, acoustic_token_embeds, text_embeds[:, 1:, :]], dim=1)
            
            # # ✨✨✨【关键修改 2】: 构建并传入与 inputs_embeds 匹配的 attention_mask ✨✨✨
            # # 原始文本的 attention_mask
            # text_attention_mask = batch['attention_mask']
            # # 为新插入的2个多模态token创建mask (batch_size, 2)
            # extra_tokens_mask = torch.ones((text_attention_mask.shape[0], 2), device=DEVICE, dtype=torch.long)
            # # 拼接成最终的 attention_mask
            # attention_mask = torch.cat([text_attention_mask[:, :1], extra_tokens_mask, text_attention_mask[:, 1:]], dim=1)

            # # 模型生成
            # outputs = model.generate(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=attention_mask, # 传入 attention_mask
            #     max_new_tokens=20,
            #     pad_token_id=tokenizer.eos_token_id,
            #     eos_token_id=tokenizer.eos_token_id,
            #     do_sample=True, temperature=0.7, top_p=0.9
            # )

            # prompt_lengths = [len(ids) for ids in batch['input_ids']]
            # generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
            # responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

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