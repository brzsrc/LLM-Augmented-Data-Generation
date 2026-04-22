# AutoDL 部署指南：Qwen3-8B 标注 HeartSteps User 2

## 一、租机器

1. 登录 [AutoDL](https://www.autodl.com/)
2. 选择 **A100-SXM4-80GB**（约 2.6 元/小时）或 **A100-PCIE-40GB**（约 1.8 元/小时）
3. 镜像选择：**PyTorch 2.1.0 + Python 3.10 + CUDA 12.1**
4. 系统盘 30GB（默认），数据盘 50GB（存模型）

## 二、上传文件

SSH 连接后，上传以下 4 个文件到 `/root/data/`：

```bash
mkdir -p /root/data /root/output

# 方式一：scp 上传（从本地）
scp suggestions.csv users.csv run_user2_annotation.py root@<你的IP>:/root/data/

# 方式二：用 AutoDL 的 JupyterLab 文件管理器上传
```

需要上传的文件：
- `suggestions.csv`（HeartSteps V1 数据）
- `users.csv`（用户画像数据）
- `run_user2_annotation.py`（标注脚本）

## 三、安装依赖 + 下载模型

```bash
# 安装 vLLM（约 2 分钟）
pip install vllm pandas numpy scipy -q

# 设置 HuggingFace 镜像（AutoDL 国内网络）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型到数据盘（约 4.5GB，3-5 分钟）
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B-AWQ', local_dir='/root/models/Qwen3-8B-AWQ')
print('Done!')
"
```

## 四、运行标注

```bash
cd /root/data
python3 run_user2_annotation.py
```

### 运行过程

脚本会：
1. 加载 suggestions.csv，提取 User 2 的数据（约 210 个决策点）
2. 从 users.csv 构建 User 2 的英文画像
3. 加载 Qwen3-8B-AWQ（约 1-2 分钟）
4. 对每个决策点运行：
   - 重要性评分（1次调用）
   - 推送接受度评分（5次采样）
   - 情境可行性评分（5次采样）
   - 心理能量评分（5次采样）
   - 约每10个点触发一次反思（额外 4 次调用）

每 10 个点打印一次进度，包含耗时和 ETA。

### 预计时间和成本

| GPU | 预计时间 | 预计成本 |
|-----|---------|---------|
| A100-80GB | 1-2 小时 | 3-5 元 |
| A100-40GB | 2-3 小时 | 4-6 元 |

### 断点续跑

如果中途断开，重新运行脚本会检测到 checkpoint 并询问是否续跑：

```
Found checkpoint: 80/210 points completed
Resume from checkpoint? (y/n): y
```

## 五、获取结果

标注完成后，输出文件在 `/root/output/`：

```bash
# 主要结果文件
ls /root/output/
# user2_annotated.csv      - 标注结果（在原数据基础上增加了6列）
# user2_memory_log.json    - 记忆流完整日志
# user2_checkpoint.json    - 断点文件

# 下载到本地
scp root@<你的IP>:/root/output/user2_annotated.csv .
```

### 新增列说明

| 列名 | 说明 | 范围 |
|------|------|------|
| `receptivity_raw` | 推送接受度（原始） | 1-5 |
| `feasibility_raw` | 情境可行性（原始） | 1-5 |
| `energy_raw` | 心理能量（原始） | 1-5 |
| `receptivity` | 推送接受度（时序平滑后） | 1-5 |
| `feasibility` | 情境可行性（时序平滑后） | 1-5 |
| `energy` | 心理能量（时序平滑后） | 1-5 |

## 六、下一步

拿到 `user2_annotated.csv` 后：
1. 检查标注质量（评分分布是否合理，是否和行为数据一致）
2. 如果质量 OK，修改脚本对全部 37 人运行（预计 A100-80GB 约 40-70 小时，~100-180 元）
3. 或者分批运行：每次跑 5-10 个用户，用断点续跑功能

## 七、常见问题

**Q: 模型下载失败？**
```bash
# 检查是否设置了镜像
echo $HF_ENDPOINT
# 应该输出 https://hf-mirror.com

# 如果还不行，用 modelscope
pip install modelscope
python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-8B-AWQ', cache_dir='/root/models/Qwen3-8B-AWQ')
"
```

**Q: CUDA out of memory？**
```bash
# 在脚本中降低 gpu_memory_utilization
# 找到 Qwen3BLLM.__init__ 中的这行：
gpu_memory_utilization=0.90
# 改为：
gpu_memory_utilization=0.80
```

**Q: 想用更便宜的 GPU？**
- RTX 4090 (24GB)：可以跑，但慢约 2-3 倍
- RTX 3090 (24GB)：可以跑，约 3-4 倍
- T4 (16GB)：勉强可以跑 AWQ 模型，约 5-8 倍
