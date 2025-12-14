import os
# 【关键】强制使用 CPU，避免加载 PyTorch CUDA 时卡死
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from safetensors.torch import load_file, save_file

def main():
    real = "_real"
    imag = "_imag"
    len = 5
    
    input_path = "model.safetensors"                   #需要转化的模型路径
    output_path = "intergrated_model.safetensors"        #转化后模型保存路径
    
    print(f"正在加载原始权重: {input_path}")

    
    try:
        # 加载文件
        state_dict = load_file(input_path, device="cpu")
        new_state_dict = {}
        
        # 遍历所有权重
        for name, tensor in state_dict.items():
            
            if name.endswith(real):
                real_tensor = tensor
                
                base_name = name[:-len]
                imag_name = base_name + imag
                if imag_name not in state_dict:
                    print(f"警告: 找不到 {name} 对应的虚部，将跳过此层！")
                    continue
                
                else:
                    imag_tensor = state_dict[imag_name]
                #weight_real,weight_imag,weight
                
                # [output dim,input dim]
                    complex_tensor = torch.cat([real_tensor, imag_tensor], dim=0)
                
                    new_state_dict[base_name] = complex_tensor
                
                    print(f"处理复数层: {base_name} | 形状: {complex_tensor.shape}")
                
            
                
            
            elif not name.endswith(real) and not name.endswith(imag):
                new_state_dict[name] = tensor

        print(f"处理完成，正在保存到: {output_path}")
        save_file(new_state_dict, output_path)
        print("成功！")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()