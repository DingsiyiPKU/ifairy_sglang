# 这里不导入 torch，只导入轻量级的 safe_open，避免卡死
from safetensors import safe_open

print("begin")

# 你的文件路径
weights_path = "/home/wangyuxing/bitnet_models/model.safetensors"

try:
    print(f"正在打开文件: {weights_path}")
    
    # device="cpu" 强制使用CPU读取，不触碰显卡
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        print("文件打开成功！准备读取键值...")
        
        # 获取所有参数的名字
        keys = f.keys()
        print(f"共检测到 {len(keys)} 个张量。\n")

        for i, key in enumerate(keys):
            # 获取具体的张量数据
            tensor = f.get_tensor(key)
            
            print(f"[{i}] {key}")
            print(f"    Shape: {tensor.shape}")
            print(f"    Dtype: {tensor.dtype}")
            print("-" * 30)
            
            if i >= 9: # 只看前10个
                break

except FileNotFoundError:
    print("错误：找不到文件，请检查路径是否正确。")
except Exception as e:
    print(f"发生错误: {e}")