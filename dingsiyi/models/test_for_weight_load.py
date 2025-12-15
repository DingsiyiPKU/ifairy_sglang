import os
import torch
from safetensors.torch import load_file
from transformers import AutoConfig

# 假设 qwen2.py 在同一个目录下
from  bitnet_sglang import BitNetForCausalLM

def load_hf_weights_into_sglang_model(model_path: str):

    
    print("--- 开始加载模型 ---")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"  - 正在从 '{model_path}' 加载 config.json...")
    config = AutoConfig.from_pretrained(model_path)

    print("  - 正在创建 SGLang Qwen2 模型架构...")
    # 使用 with torch.device(device) 确保模型在创建时就在 GPU 上，避免额外的数据拷贝
    with torch.device(device):
        model = BitNetForCausalLM(config).to(dtype)

    # 4. 加载权重文件到 CPU
    weight_file = os.path.join(model_path, "model.safetensors")
    print(f"  - 正在从 '{weight_file}' 加载权重...")
    

    state_dict = load_file(weight_file, device="cpu")

    print("  - 正在将权重映射并加载到 SGLang 模型中...")

    model.load_weights(state_dict.items())

    print("\n--- 模型加载成功！ ---")
    
    # 打印一些信息以验证
    print(f"模型已加载到设备: {model.device}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params / 1e9:.2f}B") # 应该约等于 1.54B
    
    return model

if __name__ == "__main__":
    hf_model_path = "/path/to/your/downloaded/Qwen2-1.5B-Instruct"
    
    if not os.path.exists(hf_model_path) or "Qwen2" not in hf_model_path:
        print(f"错误：模型路径 '{hf_model_path}' 不存在或不正确。")
        print("请下载 Qwen2-1.5B-Instruct 模型并更新路径。")
    else:
        sglang_model = load_hf_weights_into_sglang_model(hf_model_path)
        
        print("\n现在你可以将 'sglang_model' 用于 SGLang 的推理后端了。")