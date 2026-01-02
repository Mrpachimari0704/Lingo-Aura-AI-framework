# # ==============================================================================
# #  evaluate.py (Lingo-Aura on CMU-MOSEI - Inference & Evaluation)
# # ==============================================================================
# #
# # HOW TO RUN:
# # 1. 确保训练已完成，并且模型权重已保存在 MODEL_PATH 指定的目录中。
# # 2. 确保所有依赖项、数据文件和 prompt 模板都与训练时相同。
# # 3. 直接运行: python evaluate.py

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# import re
# import json
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn as nn


# # 导入我们项目中的核心组件
# # 假设 evaluate.py 与 lingo_aura_standalone.py 在同一目录
# from train_full_model_lora_r16_normalize import  MOSEIDataset
# from mmsdk import mmdatasdk as md
# from peft import PeftModel
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# class Config:
#     DATA_PATH = "./data/cmumosei/"
#     COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
#     PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
#     OUTPUT_DIR = "output/all_model_LoRA_attention_right_label_r16_normalize_visionpre_changeforward"

#     LLM_NAME = "./phi-2"
#     VISUAL_FEATURE_DIM = 35      # CMU_MOSEI_VisualFacet42 的特征维度
#     ACOUSTIC_FEATURE_DIM = 74    # CMU_MOSEI_COVAREP 的特征维度

#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     EPOCHS = 5
#     BATCH_SIZE = 16
#     LEARNING_RATE = 2e-5
#     # LEARNING_RATE = 1e-4
#     # GRADIENT_A

# # --- 1. 推理专用数据处理 ---
# # class MOSEIEvaluationDataset(MOSEIDataset):
# #     pass
    

# def create_evaluation_collate_fn(tokenizer, prompt_template):
#     def collate_fn(batch):
#         # 从模板中分离出 human 部分
#         human_template, _ = prompt_template.split("### Assistant:")
#         human_template += "### Assistant:"

#         prompts = []
#         for idx, item in enumerate(batch):
#             # 填充 Human 部分的模板，作为模型的输入
#             prompt = human_template.format(
#                 information_stance=item['cognitive_label'].get("Information Stance", ["N/A"])[0],
#                 reasoning_mode=item['cognitive_label'].get("Reasoning Mode", ["N/A"])[0],
#                 transcription=item['text']
#             ).strip()
#             prompts.append(prompt)
            
#             # 检查Prompt是否为空
#             if not prompt:
#                 print(f"警告：样本{idx}的Prompt为空！item详情：{item}")
        
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         # 只对 prompt 部分进行分词
#         tokenized = tokenizer(prompts, padding='longest', return_tensors="pt")

#         return {
#             'input_ids': tokenized['input_ids'],
#             'attention_mask': tokenized['attention_mask'],
#             'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True),
#             'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True),
#             'ground_truth_scores': torch.tensor([item['emotion_score'] for item in batch])
#         }
#     return collate_fn


# # class LingoAuraInferenceModel(nn.Module):
# #     def __init__(self, config, tokenizer, base_model, visual_projector, acoustic_projector, visual_attention, acoustic_attention, multimodal_gate):
# #         super().__init__()
# #         self.config = config
# #         self.tokenizer = tokenizer
# #         self.base_model = base_model 
# #         self.visual_projector = visual_projector
# #         self.acoustic_projector = acoustic_projector
# #         self.visual_attention = visual_attention
# #         self.acoustic_attention = acoustic_attention
# #         self.multimodal_gate = multimodal_gate

# #     def forward(self, input_ids, attention_mask, visual_features, acoustic_features):
# #         # ✨✨✨【终极核心修改】: 我们完全绕开猴子补丁，自己完成第一次前向传播 ✨✨✨
# #         # 校验input_ids长度，防止空输入
# #         if input_ids.size(1) == 0:
# #             raise ValueError("input_ids为空，请检查Prompt生成逻辑！")
# #         # 1. 正常进行一次纯文本的前向传播，获取所有隐藏层
# #         with torch.no_grad(): # 确保这部分不计算梯度
# #             outputs_text_only = self.base_model(
# #                 input_ids=input_ids,
# #                 attention_mask=attention_mask,
# #                 output_hidden_states=True,
# #                 return_dict=True
# #             )
        
# #         # 2. 准备多模态修正信号
# #         last_hidden_state = outputs_text_only.hidden_states[-1]
        
# #         projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
# #         projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
        
# #         last_layer_query = last_hidden_state[:, 0:1, :].to(torch.bfloat16)
        
# #         visual_token, _ = self.visual_attention(query=last_layer_query, key=projected_visual, value=projected_visual)
# #         acoustic_token, _ = self.acoustic_attention(query=last_layer_query, key=projected_acoustic, value=projected_acoustic)
        
# #         multimodal_correction = (visual_token + acoustic_token)
        
# #         # 3. 将修正信号注入到所有隐藏层
# #         modified_hidden_states = list(outputs_text_only.hidden_states[1:])
# #         for i in range(len(modified_hidden_states)):
# #             correction_broadcasted = multimodal_correction.expand_as(modified_hidden_states[i])
# #             modified_hidden_states[i] = modified_hidden_states[i] + self.multimodal_gate * correction_broadcasted
        
# #         final_last_hidden_state = modified_hidden_states[-1]
        
# #         # 4. 用修正后的隐藏状态，重新计算 past_key_values
# #         # 这是让 generate 函数能够从我们的修正点开始续写的关键
# #         # 注意: 这个逻辑依赖于 Phi-2 的内部实现，但通常是可行的
# #         # 我们需要一个新的前向传播来生成修正后的 past_key_values
# #         with torch.no_grad():
# #             outputs_modified = self.base_model(
# #                 inputs_embeds=final_last_hidden_state, # ✨ 用修正后的隐藏状态作为输入
# #                 attention_mask=attention_mask,
# #                 use_cache=True # 必须开启 use_cache
# #             )
# #             past_key_values = outputs_modified.past_key_values

# #         # ✨✨✨【核心修复】: 调整 generate 函数的输入 ✨✨✨
        
# #         # 1. 确定续写的“起点”
# #         #    当提供了 past_key_values 时，我们只需要提供最后一个 prompt token 作为新输入的开始。
# #         #    它的形状应该是 (batch_size, 1)。
# #         next_token_ids = input_ids[:, -1:]

# #         # 2. 调用 generate 函数，但只传入“起点” token，而不是完整的 input_ids
# #         outputs = self.base_model.generate(
# #             input_ids=next_token_ids,       # <--- 核心修改
# #             attention_mask=attention_mask,  # attention_mask 仍然需要，用来确定总长度
# #             past_key_values=past_key_values,
# #             max_new_tokens=20,
# #             pad_token_id=self.tokenizer.eos_token_id,
# #             eos_token_id=self.tokenizer.eos_token_id,
# #             do_sample=False,
# #             min_new_tokens=1,
# #         )
# #         return outputs



# class LingoAuraLLM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#         ###量化配置
#         quant_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#         )
#         self.llm = AutoModelForCausalLM.from_pretrained(
#             config.LLM_NAME,
#             quantization_config=quant_config, # <-- 明确指定使用 float16 半精度
#             device_map=config.DEVICE,
#             trust_remote_code=True
#         )
#         self.llm = prepare_model_for_kbit_training(self.llm)
#         # --- d. 定义LoRA配置 ---
#         # 对于 Phi-2, 常见的 target_modules 是 "q_proj", "k_proj", "v_proj", "dense"
#         lora_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             r=16,
#             lora_alpha=32,
#             target_modules=["q_proj", "k_proj", "v_proj", "dense"], 
#             lora_dropout=0.1,
#             bias="none",
#         )
#         # --- e. 将LoRA适配器应用到LLM上 ---
#         self.llm = get_peft_model(self.llm, lora_config)

#         # self.llm = AutoModelForCausalLM.from_pretrained(config.LLM_NAME, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
#         llama_hidden_size = self.llm.config.hidden_size
#         self.visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
#         self.acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)
       
       
#         # ✨✨✨【核心修改 1】: 定义跨模态注意力模块 ✨✨✨
#         # 我们使用一个简单的单头注意力，头数(num_heads)可以调整
#         self.visual_attention = nn.MultiheadAttention(
#             embed_dim=llama_hidden_size, num_heads=4, batch_first=True
#         )
#         self.acoustic_attention = nn.MultiheadAttention(
#             embed_dim=llama_hidden_size, num_heads=4, batch_first=True
#         )

#         # 我们让projectors跟随LLM的设备
#         llm_device = self.llm.device
        
#         # 3. 将 projectors 移动到与LLM相同的设备上
#         self.visual_projector.to(llm_device, dtype=torch.bfloat16)
#         self.acoustic_projector.to(llm_device, dtype=torch.bfloat16)

#         # 将注意力模块也移动到正确的设备和类型
#         self.visual_attention.to(llm_device, dtype=torch.bfloat16)
#         self.acoustic_attention.to(llm_device, dtype=torch.bfloat16)


#         # ✨✨✨【新的模块】: 定义一个 Adapter 门控 ✨✨✨
#         # 这个门控决定了多模态信息以多大的“音量”被注入
#         self.multimodal_gate = nn.Parameter(torch.tensor([0.1])).to(llm_device, dtype=torch.bfloat16) # 初始值设为一个较小的值

#     def forward(self, input_ids, attention_mask, labels, visual_features, acoustic_features):
#         # 1. 像以前一样，准备好多模态 token
#         projected_visual = self.visual_projector(visual_features.to(torch.bfloat16))
#         projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.bfloat16))
        
#         # 注意：我们这里不再需要 text_embeds
#         # 我们先让模型正常处理文本
        
#         # 2. 【第一步】: 让 LLM 正常地进行一次纯文本的前向传播
#         # 我们需要获取每一层的隐藏状态
#         outputs_text_only = self.llm(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True, # 关键：获取所有中间层的输出
#             return_dict=True
#         )
        
#         # hidden_states 是一个元组，包含了输入嵌入和每一层 Transformer 的输出
#         # hidden_states[0] 是输入嵌入, hidden_states[1] 是第一层输出, ...
#         text_hidden_states = outputs_text_only.hidden_states

#         # 3. 【第二步】: 准备多模态“修正信号”
#         # 我们用最后一层的文本隐藏状态的第一个 token (<s>) 作为 query
#         last_layer_query = text_hidden_states[-1][:, 0:1, :].to(torch.bfloat16)
        
#         visual_token, _ = self.visual_attention(
#             query=last_layer_query, key=projected_visual, value=projected_visual
#         )
#         acoustic_token, _ = self.acoustic_attention(
#             query=last_layer_query, key=projected_acoustic, value=projected_acoustic
#         )
        
#         # 将两个多模态token相加，形成一个统一的“修正信号”
#         multimodal_correction = (visual_token + acoustic_token)

#         # 4. 【第三步】: 将“修正信号”注入到每一层
#         # 我们从第二层开始注入 (跳过输入嵌入层)
#         modified_hidden_states = list(text_hidden_states[1:]) # 取出所有 transformer 层的输出
        
#         for i in range(len(modified_hidden_states)):
#             # 将修正信号“广播”到和文本序列一样的长度
#             # 然后通过一个可学习的门控，“加”到每一层的隐藏状态上
#             correction_broadcasted = multimodal_correction.expand_as(modified_hidden_states[i]).to(self.llm.device, dtype=torch.bfloat16) # 初始值设为一个较小的值
#             modified_hidden_states[i] = modified_hidden_states[i] + self.multimodal_gate * correction_broadcasted

#         # 5. 【第四步】: 将被修正过的最后一层隐藏状态，送入最终的 LM Head 计算损失
#         # 这是最关键的一步，我们需要手动完成 CausalLM 的最后一步
#         last_hidden_state = modified_hidden_states[-1]
#         logits = self.llm.lm_head(last_hidden_state)
        
#         # 手动计算交叉熵损失
#         loss_fct = nn.CrossEntropyLoss()
        
#         # 将 logits 和 labels 的形状对齐
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
        
#         # Flatten the tokens
#         loss = loss_fct(shift_logits.view(-1, self.llm.config.vocab_size), shift_labels.view(-1))
        
#         return loss
    

# # --- 主评估函数 ---
# def evaluate():
#     print("="*60)
#     print("Lingo-Aura LLM - CMU-MOSEI 模型评估脚本")
#     print("="*60)

#     config = Config()
#     MODEL_PATH = config.OUTPUT_DIR 
#     DEVICE = config.DEVICE

#     # --- [1/4] 模型加载部分（已修改） ---
#     print(f"\n[1/4] 正在从 '{MODEL_PATH}' 加载模型...")

#     # a. 在加载基础模型时，就指定好最终的 torch_dtype
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )
#     base_model = AutoModelForCausalLM.from_pretrained(
#         config.LLM_NAME,
#         quantization_config=quant_config,
#         device_map=config.DEVICE,
#         trust_remote_code=True,
#         # ✨✨✨【关键修改 1】✨✨✨
#         # 在加载时就明确指定计算和权重的dtype，防止后续转换
#         torch_dtype=torch.bfloat16, 
#     )
#     tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True')
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # b. 将LoRA适配器融合到基础模型中
#     model = PeftModel.from_pretrained(base_model, MODEL_PATH)
#     model = model.merge_and_unload()
#     print(" - LoRA 适配器已加载并融合。")

#     # c. 创建并加载 Projectors
#     llama_hidden_size = model.config.hidden_size

#     #开始重建架构 ✨✨✨ ---
#     # 1. 创建 Projector 模块的“空壳”
#     visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
#     acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)

#     # 2. 创建 Attention 模块的“空壳”
#     #    这里的参数（embed_dim, num_heads）必须与训练时完全一致！
#     visual_attention = nn.MultiheadAttention(embed_dim=llama_hidden_size, num_heads=4, batch_first=True)
#     acoustic_attention = nn.MultiheadAttention(embed_dim=llama_hidden_size, num_heads=4, batch_first=True)

#     # --- ✨✨✨ 加载权重 ✨✨✨ ---

#     # 3. 加载 Projector 的权重
#     visual_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "visual_projector.pt")))
#     acoustic_projector.load_state_dict(torch.load(os.path.join(MODEL_PATH, "acoustic_projector.pt")))

#     # 4. 加载 Attention 模块的权重
#     visual_attention.load_state_dict(torch.load(os.path.join(MODEL_PATH, "visual_attention.pt")))
#     acoustic_attention.load_state_dict(torch.load(os.path.join(MODEL_PATH, "acoustic_attention.pt")))

#     # --- ✨✨✨ 移动设备和类型，并“挂载”到主模型上 ✨✨✨ ---

#     llm_device = next(model.parameters()).device
#     visual_projector.to(device=llm_device, dtype=torch.bfloat16)
#     acoustic_projector.to(device=llm_device, dtype=torch.bfloat16)
#     visual_attention.to(device=llm_device, dtype=torch.bfloat16)
#     acoustic_attention.to(device=llm_device, dtype=torch.bfloat16)

#     print(" - 警告: 未找到 'multimodal_gate.pt' 文件。将手动设置门控参数值。")
#     gate_value_to_test = 0.1 
#     # ✨ 新增：加载 multimodal_gate 参数 ✨
#     # multimodal_gate = nn.Parameter(torch.zeros(1)) # 创建一个空的 Parameter
#     # multimodal_gate.load_state_dict({'weight': torch.load(os.path.join(MODEL_PATH, "multimodal_gate.pt"))['weight'].to(llm_device)})
    
#     multimodal_gate = nn.Parameter(torch.tensor([gate_value_to_test])).to(llm_device, dtype=torch.bfloat16)
#     print(f"   - 手动设置门控值为: {multimodal_gate.item():.4f}")

#    # --- 初始化我们的推理封装类 (现在非常直接) ---
#     inference_model = LingoAuraInferenceModel(
#         config=config, tokenizer=tokenizer, base_model=model,
#         visual_projector=visual_projector, acoustic_projector=acoustic_projector,
#         visual_attention=visual_attention, acoustic_attention=acoustic_attention,
#         multimodal_gate=multimodal_gate
#     )
#     inference_model.eval()
#     print(" - 多模态推理模型已初始化完成。")
    

#     print(f"\n[2/4] 正在加载 CMU-MOSEI 测试数据集...")

#     # ✨✨✨ 2. 加载保存的统计量 ✨✨✨
#     stats_path = os.path.join(config.OUTPUT_DIR, 'normalization_stats.json')
#     try:
#         with open(stats_path, 'r') as f:
#             stats = json.load(f)
#         visual_mean = torch.tensor(stats['visual_mean'])
#         visual_std = torch.tensor(stats['visual_std'])
#         acoustic_mean = torch.tensor(stats['acoustic_mean'])
#         acoustic_std = torch.tensor(stats['acoustic_std'])
        
#         visual_stats = (visual_mean, visual_std)
#         acoustic_stats = (acoustic_mean, acoustic_std)
#         print(" - 成功加载归一化统计量。")
#     except FileNotFoundError:
#         print(f"警告: 找不到归一化文件 {stats_path}。将不进行归一化处理。")
#         visual_stats = None
#         acoustic_stats = None


#     cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
#     with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
#         prompt_template = f.read()
    
#     # ✨✨✨ 3. 将统计量传入测试集 Dataset ✨✨✨
#     test_dataset = MOSEIDataset(
#         cognitive_df, 
#         md.cmu_mosei.standard_folds.standard_test_fold, 
#         prompt_template,
#         visual_stats=visual_stats,
#         acoustic_stats=acoustic_stats
#     )
#     collate_fn = create_evaluation_collate_fn(tokenizer, prompt_template)
#     test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, collate_fn=collate_fn, num_workers=4)
    
    
#     print(f"\n[3/4] 正在测试集上进行推理...")
#     all_predictions = []
#     all_ground_truths = []

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Evaluating on Test Set"):

#             # 从batch中提取参数（确保设备一致）
#             input_ids = batch['input_ids'].to(DEVICE)
#             attention_mask = batch['attention_mask'].to(DEVICE)
#             visual_features = batch['visual_features'].to(DEVICE)
#             acoustic_features = batch['acoustic_features'].to(DEVICE)
#             ground_truths = batch['ground_truth_scores']

#             # --- ✨ 调用整合后的模型生成结果 ✨ ---
#             outputs = inference_model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 visual_features=visual_features,
#                 acoustic_features=acoustic_features
#             )

#             # 后续的token解码、分数提取逻辑保持不变
#             prompt_lengths = [len(ids) for ids in input_ids]
#             generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
#             responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

#             # ✨✨✨ 添加这行打印语句 ✨✨✨
#             print("原始生成内容:", responses) 
            
#             for res in responses:
#                 match = re.search(r"[-+]?\d+(?:\.\d+)?", res)
#                 if match:
#                     pred_score = float(match.group())
#                     all_predictions.append(pred_score)
#                 else:
#                     all_predictions.append(0.0)

#             all_ground_truths.extend(batch['ground_truth_scores'].cpu().numpy())

#     print(f"\n[4/4] 计算评估指标...")
#     print("\n" + "="*20 + " 最终评估结果 " + "="*20)
#     print("模型的所有预测值:", all_predictions[:50]) # 打印前50个看看
#     gts = np.array(all_ground_truths)
#     preds = np.array(all_predictions)

#     mae = np.mean(np.abs(gts - preds))
#     print(f"平均绝对误差 (MAE) ↓: {mae:.4f}")

#     acc2 = accuracy_score(gts >= 0, preds >= 0)
#     print(f"二元准确率 (Acc-2) ↑: {acc2*100:.2f}%")

#     f1 = f1_score(gts >= 0, preds >= 0, average='weighted')
#     print(f"加权F1分数 (F1-Score) ↑: {f1:.4f}")

#     # 修改后
#     valid_indices = ~np.isnan(gts) & ~np.isnan(preds)
#     gts_valid = gts[valid_indices]
#     preds_valid = preds[valid_indices]

#     # ✨✨✨ 核心修复：检查标准差是否为0 ✨✨✨
#     if gts_valid.size > 1 and preds_valid.size > 1 and \
#     np.std(gts_valid) > 0 and np.std(preds_valid) > 0:
        
#         corr = np.corrcoef(gts_valid, preds_valid)[0, 1]
#         print(f"皮尔逊相关系数 (Corr) ↑: {corr:.4f}")
#     else:
#         print("无法计算相关系数 (原因: 预测值或真实值的标准差为0，或有效数据点不足)。")


# if __name__ == "__main__":
#     evaluate()



import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from mmsdk import mmdatasdk as md
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Config:
    DATA_PATH = "./data/cmumosei/"
    COGNITIVE_LABELS_CSV = os.path.join(DATA_PATH, "cmu_mosei_with_cognitive_labels_v4.csv")
    PROMPT_TEMPLATE_PATH = "./prompts/cognitive_informed_prompt.txt"
    OUTPUT_DIR = "output/all_model_LoRA_attention_right_label_r16_normalize_visionpre_changeforward"

    LLM_NAME = "./phi-2"
    VISUAL_FEATURE_DIM = 35      # 与训练保持一致
    ACOUSTIC_FEATURE_DIM = 74    # 与训练保持一致

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8  # 推理批次大小可调整
    MAX_NEW_TOKENS = 20  # 生成文本最大长度


class MOSEIEvaluationDataset:
    def __init__(self, cognitive_df, split_ids, prompt_template, 
                 visual_stats=None, acoustic_stats=None):
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
        
        self.raw_data  = self._prepare_data()

        # 应用归一化
        if visual_stats and acoustic_stats:
            self.visual_mean, self.visual_std = visual_stats
            self.acoustic_mean, self.acoustic_std = acoustic_stats
            self.data = self._normalize_data(self.raw_data)
        else:
            raise ValueError("Evaluation dataset requires stats from the training set.")

    def _load_csd_files(self):
        recipe = {
            self.visual_field: os.path.join(Config.DATA_PATH, self.visual_field + '.csd'),
            self.acoustic_field: os.path.join(Config.DATA_PATH, self.acoustic_field + '.csd'),
            self.label_field: os.path.join(Config.DATA_PATH, self.label_field + '.csd'),
        }
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
                
                prepared_data.append({
                    'visual': torch.from_numpy(visual),
                    'acoustic': torch.from_numpy(acoustic),
                    'text': text,
                    'emotion_score': emotion_score,
                    'cognitive_label': cognitive_label,
                    'sample_id': segment_id
                })
            except Exception:
                skipped_count += 1
                continue
        if skipped_count > 0:
            print(f"警告: 推理数据准备中跳过 {skipped_count} 个样本。")
        return prepared_data

    def _normalize_data(self, raw_data_list):
        normalized_data = []
        for item in raw_data_list:
            normalized_item = item.copy()
            normalized_item['visual'] = (item['visual'] - self.visual_mean) / self.visual_std
            normalized_item['acoustic'] = (item['acoustic'] - self.acoustic_mean) / self.acoustic_std
            normalized_data.append(normalized_item)
        return normalized_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class LingoAuraInferenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型（与训练量化配置一致）
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_NAME,
            quantization_config=quant_config,
            device_map=config.DEVICE,
            trust_remote_code=True,
            dtype=torch.float32
        )

        # 融合LoRA权重
        self.llm = PeftModel.from_pretrained(self.llm, config.OUTPUT_DIR)
        self.llm = self.llm.merge_and_unload()
        self.llm.eval()

        # 重建多模态组件（与训练结构完全一致）
        llama_hidden_size = self.llm.config.hidden_size
        self.visual_projector = nn.Linear(config.VISUAL_FEATURE_DIM, llama_hidden_size)
        self.acoustic_projector = nn.Linear(config.ACOUSTIC_FEATURE_DIM, llama_hidden_size)
        
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=llama_hidden_size, num_heads=4, batch_first=True
        )
        self.acoustic_attention = nn.MultiheadAttention(
            embed_dim=llama_hidden_size, num_heads=4, batch_first=True
        )
        


        # 加载多模态组件权重
        self._load_multimodal_weights()

        # 设备统一
        llm_device = self.llm.device
        print(f"LLM 设备: {llm_device}")  # 需输出类似 'cuda:0'
        self.visual_projector.to(llm_device, dtype=torch.float32)
        self.acoustic_projector.to(llm_device, dtype=torch.float32)
        self.visual_attention.to(llm_device, dtype=torch.float32)
        self.acoustic_attention.to(llm_device, dtype=torch.float32)
        self.multimodal_gate = nn.Parameter(
            torch.tensor([0.1], device=llm_device, dtype=torch.float32)  # 直接指定设备和类型
        )        
        print("llm_device",llm_device)

    def _load_multimodal_weights(self):
        """加载训练时保存的多模态组件权重"""
        weight_dir = self.config.OUTPUT_DIR
        try:
            self.visual_projector.load_state_dict(torch.load(os.path.join(weight_dir, "visual_projector.pt")))
            self.acoustic_projector.load_state_dict(torch.load(os.path.join(weight_dir, "acoustic_projector.pt")))
            self.visual_attention.load_state_dict(torch.load(os.path.join(weight_dir, "visual_attention.pt")))
            self.acoustic_attention.load_state_dict(torch.load(os.path.join(weight_dir, "acoustic_attention.pt")))
            
            # 加载门控权重（训练时未显式保存，若训练时有保存逻辑需对应修改）
            # 若训练时未保存，可保持初始值或添加保存逻辑
            print("多模态组件权重加载成功")
        except Exception as e:
            print(f"权重加载警告: {str(e)}，将使用初始值")

    def forward(self, input_ids, attention_mask, visual_features, acoustic_features):
        print("\n===== 输入张量检查 =====")
        print(f"input_ids 形状: {input_ids.shape} (batch_size, seq_len)")  # 应类似 (8, 128)
        print(f"input_ids 非空: {input_ids.numel() > 0}")  # 应返回 True
        print(f"attention_mask 形状: {attention_mask.shape}")  # 应与 input_ids 一致
        print(f"attention_mask 有效长度: {attention_mask.sum(dim=1)}")  # 每个样本的有效token数，不应全为0
        print("=======================\n")


        # 1. 多模态特征投影（保持不变）
        projected_visual = self.visual_projector(visual_features.to(torch.float32))
        projected_acoustic = self.acoustic_projector(acoustic_features.to(torch.float32))

        # 2. 纯文本前向传播获取隐藏状态（保持不变）
        with torch.no_grad():
            outputs_text_only = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        text_hidden_states = outputs_text_only.hidden_states
        text_hidden_states = [h.to(torch.float32) for h in text_hidden_states]  # 确保类型统一

        # 3. 计算多模态修正信号（保持不变）
        last_layer_query = text_hidden_states[-1][:, 0:1, :].to(torch.float32)
        
        visual_token, _ = self.visual_attention(
            query=last_layer_query, key=projected_visual, value=projected_visual
        )
        acoustic_token, _ = self.acoustic_attention(
            query=last_layer_query, key=projected_acoustic, value=projected_acoustic
        )
        multimodal_correction = (visual_token + acoustic_token).to(torch.float32)

        # 4. 注入修正信号到各层（保持不变）
        modified_hidden_states = list(text_hidden_states[1:])
        for i in range(len(modified_hidden_states)):
            correction_broadcasted = multimodal_correction.expand_as(modified_hidden_states[i]).to(torch.float32)
            modified_hidden_states[i] = modified_hidden_states[i] + self.multimodal_gate * correction_broadcasted

        # 5. 修正生成逻辑：直接使用修正后的隐藏状态和原始input_ids生成
        final_hidden = modified_hidden_states[-1]
        # 5. 生成past_key_values时，使用完整的attention_mask
        
        with torch.no_grad():
            outputs_modified = self.llm(
                inputs_embeds=final_hidden,
                attention_mask=attention_mask,  # 保持与inputs_embeds长度一致（351）
                use_cache=True
            )
            past_key_values = outputs_modified.past_key_values

        # # 生成时仅传入最后一个token作为新输入
        outputs = self.llm.generate(
            inputs_embeds=final_hidden,
            attention_mask=attention_mask,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
            min_new_tokens=1
        )
        
        return outputs


def create_evaluation_collate_fn(tokenizer, prompt_template):
    def collate_fn(batch):
        # 构建推理prompt
        human_template, _ = prompt_template.split("### Assistant:")
        human_template += "### Assistant:"

        prompts = []
        for item in batch:
            prompt = human_template.format(
                information_stance=item['cognitive_label'].get("Information Stance", ["N/A"])[0],
                reasoning_mode=item['cognitive_label'].get("Reasoning Mode", ["N/A"])[0],
                transcription=item['text']
            ).strip()
            prompts.append(prompt)

        # 分词处理
        tokenized = tokenizer(
            prompts,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'visual_features': pad_sequence([f['visual'] for f in batch], batch_first=True, padding_value=0.0).to(torch.float32),
            'acoustic_features': pad_sequence([f['acoustic'] for f in batch], batch_first=True, padding_value=0.0).to(torch.float32),
            'ground_truth_scores': torch.tensor([item['emotion_score'] for item in batch]),
            'sample_ids': [item['sample_id'] for item in batch]
        }
    return collate_fn


def evaluate():
    print("="*60)
    print("Lingo-Aura 推理评估 (与训练代码严格对齐)")
    print("="*60)

    config = Config()
    DEVICE = config.DEVICE

    # 1. 加载模型
    print(f"[1/4] 从 {config.OUTPUT_DIR} 加载模型...")
    model = LingoAuraInferenceModel(config)
    tokenizer = model.tokenizer
    print(f"模型加载完成，设备: {model.llm.device}")

    # 2. 加载数据和统计量
    print(f"[2/4] 加载测试数据...")
    # 加载归一化统计量
    stats_path = os.path.join(config.OUTPUT_DIR, 'normalization_stats.json')
    visual_stats, acoustic_stats = None, None
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        visual_stats = (torch.tensor(stats['visual_mean']), torch.tensor(stats['visual_std']))
        acoustic_stats = (torch.tensor(stats['acoustic_mean']), torch.tensor(stats['acoustic_std']))
        print("已加载归一化统计量")

    # 加载prompt模板
    with open(config.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # 加载测试集
    cognitive_df = pd.read_csv(config.COGNITIVE_LABELS_CSV)
    test_dataset = MOSEIEvaluationDataset(
        cognitive_df,
        md.cmu_mosei.standard_folds.standard_test_fold,
        prompt_template,
        visual_stats=visual_stats,
        acoustic_stats=acoustic_stats
    )
    collate_fn = create_evaluation_collate_fn(tokenizer, prompt_template)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4
    )

    # 3. 推理过程
    print(f"[3/4] 开始推理...")
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="推理进度"):
            # 数据移至设备
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            visual_feats = batch['visual_features'].to(DEVICE, dtype=torch.float32)
            acoustic_feats = batch['acoustic_features'].to(DEVICE, dtype=torch.float32)
            gts = batch['ground_truth_scores'].numpy()


            # 新增：检查input_ids是否为空
            if input_ids.numel() == 0:
                print("警告：空的input_ids，跳过该批次")
                continue

            # 模型生成
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_feats,
                acoustic_features=acoustic_feats
            )
            
            # 解析生成结果
            prompt_lengths = [len(ids) for ids in input_ids]
            generated_tokens = [out[L:] for out, L in zip(outputs, prompt_lengths)]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # ✨✨✨ 添加这行打印语句 ✨✨✨
            print("原始生成内容:", responses) 

            # 提取分数
            for res in responses:
                match = re.search(r"[-+]?\d+\.?\d*", res)
                if match:
                    all_preds.append(float(match.group()))
                else:
                    all_preds.append(0.0)  # 解析失败时用默认值

            all_gts.extend(gts)

    # 4. 计算指标
    print(f"[4/4] 计算评估指标...")
    preds = np.array(all_preds)
    gts = np.array(all_gts)

    # 过滤无效值
    valid_mask = ~np.isnan(preds) & ~np.isnan(gts)
    preds = preds[valid_mask]
    gts = gts[valid_mask]

    if len(preds) == 0:
        print("没有有效的预测结果，无法计算指标")
        return

    # 计算指标
    mae = mean_absolute_error(gts, preds)
    acc2 = accuracy_score(gts >= 0, preds >= 0)
    f1 = f1_score(gts >= 0, preds >= 0, average='weighted')

    # 计算相关系数
    corr = 0.0
    if np.std(gts) > 1e-6 and np.std(preds) > 1e-6:
        corr = np.corrcoef(gts, preds)[0, 1]

    # 输出结果
    print("\n" + "="*30)
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"二元准确率 (Acc-2): {acc2*100:.2f}%")
    print(f"加权F1分数 (F1-Score): {f1:.4f}")
    print(f"皮尔逊相关系数 (Corr): {corr:.4f}")
    print("="*30)


if __name__ == "__main__":
    evaluate()