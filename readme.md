# Lingo-Aura: Cognitive-Informed Multimodal Sentiment Analysis
# Lingo-Aura：基于认知提示的多模态情感分析系统

## 📂 项目简介 (Project Overview)
本项目实现了一个基于大语言模型（Mistral-7B/Phi-2）的多模态情感分析框架。核心创新在于引入了**认知标签（Cognitive Labels）**作为提示，并设计了 **Double MLP + Mean Pooling** 的轻量级架构，配合 **InfoNCE 对比学习**与**分层预热（Warmup）**策略，在 CMU-MOSEI 数据集上实现了情感强度预测（Correlation）的大幅提升。

## 📦 核心交付物 (Key Deliverables)

在查看代码前，建议优先阅读以下文档，了解核心技术路径与实验结论：

1.  **`技术报告.pdf` / `技术报告.docx`**  
    📄 **[最重要]** 完整的项目技术报告。包含模型架构图、SOTA 对比、消融实验分析及最终结论。
2.  **`技术图片.pptx`**  
    📊 报告中所有架构图的可编辑源文件。

---

## 🗂️ 文件结构说明 (File Structure)

文件夹中包含多次实验迭代的脚本与日志，以下是关键文件的分类说明：

### 1. 核心代码 (Main Scripts)
这是最终验证效果最好、建议使用的版本：

*   **训练脚本 (Training)**:
    *   `train_full_model_lora_r16_normalize_Mistral7b_changeloss_doublemlp_meanpooling_contraloss0.25_Projectorwarm_dropout_singlecard.py`
    *   **说明**：这是**最终胜出方案**。集成了 Double MLP、Mean Pooling、Dropout(0.2)、对比损失(0.25权重)及分层预热策略的单卡训练脚本。
*   **推理/评估脚本 (Inference)**:
    *   `inference_all_attention_rightlabel_normalize_mistral7bmeanpooling_dropout_warm0.25.py` (需确认具体使用的推理脚本文件名，通常是配合上述训练脚本的)
    *   **说明**：包含 Few-Shot 引导与 Prefix-Forcing 策略，用于生成最终的 Acc 与 Corr 指标。
*   **数据处理**:
    *   `generate_cognitive_labels.py`: 调用 DeepSeek API 生成认知标签的脚本。


### 2. 消融实验与历史版本 (Ablation & History)
为了复现报告中的对比实验，保留了以下变体脚本：

*   `train_ablation_text_only.py`:仅文本消融实验脚本（无cognitive认知）
*   `train_full_model_lora_r16_normalize_Mistral7b.py`:无对比损失等消融实验脚本
*   `train_full_model_lora_r16_normalize_Mistral7b_changeloss_doublemlp_meanpooling_contraloss1.0_Projectorwarm_singlecard.py`:无dropout消融实验脚本
*   `..._ddp_...py`: 多显卡分布式训练版本（用于加速，但配置较复杂）。
*   `..._nolora.py`: 不使用 LoRA 的全量微调或冻结基线（用于对比）。
*   `..._noacoustic.py` / `..._novision.py`: 单模态消融实验脚本。
*   `..._contrastloss1.0...py`: 对比损失权重为 1.0 的实验版本（效果不如 0.25）。
*   `......`:

### 3. 日志与输出 (Logs & Outputs)
*   `*.out` / `*.log`: 训练过程的控制台日志记录。
*   `output/`: 模型权重（Checkpoint）、适配器（Adapter）及归一化统计量保存目录。

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
确保安装了 Python 3.12+ 及以下核心库：
```bash
pip install torch transformers peft pandas numpy tqdm mmsdk scikit-learn
```

### 2. 数据准备
请确保 `data/cmumosei/` 目录下包含以下文件：
*   CMU_MOSEI_VisualFacet42.csd
*   CMU_MOSEI_COVAREP.csd
*   CMU_MOSEI_TimestampedWords.csd
*   CMU_MOSEI_Labels.csd
*   **cmu_mosei_with_cognitive_labels_v4.csv** (由 `generate_cognitive_labels.py` 生成)

### 3. 运行训练 (Training)
使用最终推荐配置进行训练（单卡模式）：
```bash
nohup python train_full_model_lora_r16_normalize_Mistral7b_changeloss_doublemlp_meanpooling_contraloss0.25_Projectorwarm_dropout_singlecard.py > train.log 2>&1 &
```

### 4. 运行评估 (Evaluation)
加载训练好的权重进行测试：
```bash
python inference_all_attention_rightlabel_normalize_mistral7bmeanpooling_dropout_warm0.25.py
```

---

## 📊 实验结论速览

基于最终模型（Mistral-7B + Double MLP + Contrastive 0.25 + Dropout）：
*   **Accuracy (Acc-2)**: ~79.9% (与纯文本基线持平，抗噪成功)
*   **Correlation (r)**: ~0.15 (相比纯文本提升 **135%**，具备了情感强度感知能力)

详细分析请参阅 `技术报告.pdf`。