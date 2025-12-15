import os

# 【重要】如果下载速度慢或连接失败，请取消下面这行的注释使用国内镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# 模型ID
repo_id = "PKU-DS-LAB/Fairy-plus-minus-i-700M"

# 下载到当前目录下的 'models' 文件夹
local_dir = "./ifairy_models"

print(f"开始下载 {repo_id} ...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # 确保下载的是真实文件而不是快捷方式
    resume_download=True          # 支持断点续传
)
print("下载完成！")