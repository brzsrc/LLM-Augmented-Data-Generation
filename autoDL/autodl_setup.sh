#!/bin/bash
# ================================================================
# AutoDL 部署脚本：Qwen3-8B + vLLM 标注 HeartSteps User 2
# ================================================================
# 
# 使用方式：
#   1. 在 AutoDL 上租一台 A100-80GB 或 A100-40GB（约 2-3 元/小时）
#   2. 选择镜像：PyTorch 2.1+ / CUDA 12.1+
#   3. SSH 连接后运行此脚本
#
# 预计运行时间：
#   A100-80GB: ~1-2 小时（User 2 单人 210 点 × 3444 次调用）
#   A100-40GB: ~2-3 小时
#
# 预计花费：约 3-8 元人民币
# ================================================================

set -e

echo "=========================================="
echo "Step 1: 安装依赖"
echo "=========================================="

pip install vllm>=0.8.0 pandas numpy scipy --quiet

echo "=========================================="
echo "Step 2: 下载模型（首次需要，后续复用）"
echo "=========================================="
echo "模型会缓存在 ~/.cache/huggingface/"
echo "Qwen3-8B-AWQ 约 4.5GB，下载约 2-5 分钟"

# 如果 AutoDL 有 HuggingFace 镜像，用镜像加速
export HF_ENDPOINT=https://hf-mirror.com

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B-AWQ', local_dir='/root/models/Qwen3-8B-AWQ')
print('Model downloaded successfully!')
"

echo "=========================================="
echo "Step 3: 上传数据文件"
echo "=========================================="
echo "请将以下文件上传到 /root/data/ 目录："
echo "  - suggestions.csv"
echo "  - users.csv"
echo "  - run_user2_annotation.py（下面的 Python 脚本）"
echo ""
echo "可以用 scp 或 AutoDL 的文件管理界面上传"

mkdir -p /root/data /root/output

echo "=========================================="
echo "Step 4: 运行标注"
echo "=========================================="
echo "上传完文件后执行："
echo "  cd /root/data && python3 run_user2_annotation.py"
echo ""
echo "Setup complete!"
