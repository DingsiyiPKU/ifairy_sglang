import os
# 【关键】强制使用 CPU，避免加载 PyTorch CUDA 时卡死
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from safetensors.torch import load_file, save_file

def unpack_weights(packed_tensor):
    """
    BitNet 拆包逻辑：
    输入: [N, K] uint8
    输出: [4*N, K] float32 (值为 -1, 0, 1)
    
    假设压缩逻辑为：一个 uint8 包含 4 个 2-bit 的数
    映射关系通常为: 0->-1, 1->0, 2->1 (offset binary)
    """
    # 1. 转换为 int32 以便进行位运算
    w = packed_tensor.to(torch.int32)
    
    # 2. 提取 4 个 2-bit 组
    # (位移操作根据 BitNet 常见实现推断)
    p0 = (w >> 6) & 0b11
    p1 = (w >> 4) & 0b11
    p2 = (w >> 2) & 0b11
    p3 = w & 0b11
    
    # 3. 堆叠并重塑形状
    # 原形状 [640, 6912] -> 堆叠后 [4, 640, 6912] -> 展平 [2560, 6912]
    unpacked = torch.stack([p0, p1, p2, p3], dim=1).view(-1, w.shape[1])
    
    # 4. 映射数值
    # 原始 2-bit 值: 0, 1, 2, (3是填充)
    # 目标 Ternary 值: -1, 0, 1
    # 简单公式: val - 1
    unpacked_float = unpacked.to(torch.float32) - 1.0
    
    return unpacked_float

def main():
    input_path = "/home/wangyuxing/bitnet_models/model.safetensors"
    output_path = "/home/wangyuxing/bitnet_models/un_model.safetensors"
    
    print(f"正在加载原始权重: {input_path}")
    print("注意：这可能需要消耗约 5-8GB 内存，请稍候...")
    
    try:
        # 加载文件
        state_dict = load_file(input_path, device="cpu")
        new_state_dict = {}
        
        print("开始反量化处理...")
        
        # 遍历所有权重
        for name, tensor in state_dict.items():
            # 1. 如果是 Scale 参数，跳过 (它会被融合进权重里，不需要单独保存)
            if name.endswith("_scale"):
                continue
                
            # 2. 如果是 uint8 类型的权重 (这是被压缩的层)
            if tensor.dtype == torch.uint8:
                print(f"处理压缩层: {name} | 原形状: {tensor.shape}")
                
                # 寻找对应的 scale
                scale_name = name + "_scale"
                if scale_name not in state_dict:
                    print(f"警告: 找不到 {name} 对应的 scale，将跳过此层！")
                    new_state_dict[name] = tensor
                    continue
                
                scale = state_dict[scale_name].to(torch.float32)
                
                # 【第一步：解包】 形状变大 4 倍
                weights_unpacked = unpack_weights(tensor)
                
                # 【第二步：反量化】 W_real = W_ternary * scale
                weights_dequantized = weights_unpacked * scale
                
                # 转回 bfloat16 以节省显存并匹配原始精度
                new_state_dict[name] = weights_dequantized.to(torch.bfloat16)
                
                print(f"  -> 解包后形状: {new_state_dict[name].shape}")
                
            # 3. 如果是普通层 (Embedding, Norm 等，已经是 BF16)
            else:
                new_state_dict[name] = tensor

        print(f"处理完成，正在保存到: {output_path}")
        save_file(new_state_dict, output_path)
        print("成功！")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()