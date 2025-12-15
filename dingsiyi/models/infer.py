import lm_eval
from lm_eval import utils
import json

# ---------------- 配置区域 ----------------

# 1. 任务列表

TASKS = ["gsm8k", "arc_easy"] 

# 2. SGLang Server 的地址
SERVER_URL = "http://localhost:30000/v1"

# 3. 模型名称
# SGLang 启动时默认的模型名通常是路径名或者是 "default"
# 你可以通过访问 http://localhost:30000/v1/models 查看确切名字
# 这里我们假设它叫 "default"，如果报错模型不存在，请修改这里
MODEL_NAME = "default" 

# ----------------------------------------

def main():
    print(f"正在连接 SGLang Server: {SERVER_URL} ...")

    # 构建 model_args 字符串
    model_args_str = (
        f"model={MODEL_NAME},"          # 告诉 API 我要调哪个模型
        f"base_url={SERVER_URL},"       # 服务器地址
        "num_concurrent=50,"            # 并发请求数 (SGLang 吞吐高，可以开大，加速测评)
        "max_retries=3,"                # 失败重试次数
        "tokenized_requests=False"      # 必须设为 False，因为我们发的是纯文本 Prompt
    )

    # 开始测评
    results = lm_eval.simple_evaluate(
        # 核心：使用 "local-completions" 后端
        # 这意味着 lm-eval 不会加载权重，而是把 input 发送给 API
        model="local-completions", 
        
        model_args=model_args_str,
        tasks=TASKS,
        
        # device 参数在这里无效，因为计算在 Server 端
        # batch_size 参数在这里也无效，由 num_concurrent 控制并发
    )

    print("\n" + "="*50)
    print("测评完成！结果如下：")
    print("="*50)

    # 打印格式化表格
    if "results" in results:
        print(utils.make_table(results))
    
    # 也可以把结果保存到文件
    with open("sglang_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results["results"], f, indent=2, ensure_ascii=False)
        print("\n详细结果已保存至 sglang_eval_results.json")

if __name__ == "__main__":
    main()
    
    
''' 
python -m sglang.launch_server \
    --model-path /path/to/your/model \
    --port 30000 \
    --host 0.0.0.0
'''  