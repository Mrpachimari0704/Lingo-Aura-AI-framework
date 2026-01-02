from transformers import AutoModelForCausalLM

# 您的 .bin 模型文件所在的路径
model_path = "./Mistral-7B-Instruct-v0.2"
# 您希望保存 .safetensors 新模型的路径
output_path = "./Mistral-7B-Instruct-v0.2-safetensors"

# 加载 .bin 模型
# 注意：运行此脚本可能需要一个未引入此安全检查的旧版 transformers
# 或者在一个满足 torch>=2.6 的环境中运行
print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(model_path)

# 将模型保存为 safetensors 格式
print(f"Saving model to {output_path} in safetensors format...")
model.save_pretrained(output_path, safe_serialization=True)

print("Conversion complete!")