import torch
import timeit

# --- 1. 设置测试环境 ---
# 检查 CUDA 是否可用
if not torch.cuda.is_available():
    print("CUDA is not available. This benchmark requires an NVIDIA GPU.")
    exit()

# 根据您的要求，明确指定使用 GPU 0
device = 'cuda:0'
torch.cuda.set_device(device)

print(f"--- Running Benchmark on: {torch.cuda.get_device_name(device)} ({device}) ---")

# --- 2. 定义测试用的张量 ---
# 使用较大的张量来凸显性能差异
d = 512
shape = (64, 1024, 2 * d)
print(f"Tensor shape for benchmark: {shape}\n")

# 在 GPU 0 上创建张量
A = torch.randn(shape, device=device, dtype=torch.float32)
B = torch.randn(shape, device=device, dtype=torch.float32)

# --- 3. 定义要测试的三种方法 ---

def op1_cat(A, B):
    """方法 1: 拆分-计算-拼接"""
    A0, A1 = torch.chunk(A, 2, dim=-1)
    B0, B1 = torch.chunk(B, 2, dim=-1)
    New_A0 = A0 + B1
    New_A1 = A1 - B0
    return torch.cat([New_A0, New_A1], dim=-1)

def op2_transform(A, B):
    """方法 2: 变换B-然后相加 (性能较差)"""
    B0, B1 = torch.chunk(B, 2, dim=-1)
    B_transformed = torch.cat([B1, -B0], dim=-1)
    return A + B_transformed

def op3_preallocation(A, B):
    """方法 3: 预分配内存并使用 out 参数 (性能最优)"""
    result = torch.empty_like(A)
    A0, A1 = torch.chunk(A, 2, dim=-1)
    B0, B1 = torch.chunk(B, 2, dim=-1)
    res0, res1 = torch.chunk(result, 2, dim=-1)
    torch.add(A0, B1, out=res0)
    torch.subtract(A1, B0, out=res1)
    return result

# --- 4. 预热 GPU ---
# 首次 CUDA 操作有额外的启动开销，预热可以确保计时更准确
print("Warming up GPU...")
for _ in range(10):
    _ = op1_cat(A, B)
    _ = op2_transform(A, B)
    _ = op3_preallocation(A, B)
# 等待所有预热操作在 GPU 上完成
torch.cuda.synchronize()
print("Warm-up complete.\n")

# --- 5. 执行基准测试 ---
iterations = 100
print(f"Running each method for {iterations} iterations...")

t1 = timeit.timeit(lambda: op1_cat(A, B), number=iterations)
torch.cuda.synchronize()

t2 = timeit.timeit(lambda: op2_transform(A, B), number=iterations)
torch.cuda.synchronize()

t3 = timeit.timeit(lambda: op3_preallocation(A, B), number=iterations)
torch.cuda.synchronize()

# --- 6. 输出结果 ---
print("\n--- Benchmark Results ---")
print(f"Method 1 (Split-Compute-Concat): {t1:.6f} seconds")
print(f"Method 2 (Transform-Add)       : {t2:.6f} seconds")
print(f"Method 3 (Pre-allocation)      : {t3:.6f} seconds")

min_time = min(t1, t3)
print("\n--- Conclusion ---")
print(f"The pre-allocation method is the fastest.")
print(f"It is approximately {t1/t3:.2f}x faster than the Concat method.")
print(f"It is approximately {t2/t3:.2f}x faster than the slow Transform-Add method.")