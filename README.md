# PI0.5 tutorial

本仓库基于 openpi 实现了一个可复现的教程，包含：
1. `pi0.5` 在 LIBERO 上的全量微调
2. `pi0.5` 在 LIBERO 上的 LoRA 微调
3. `pi0.5` 在 xArm7 真机数据转换、训练、真机推理

## 1. Repo 结构总览


| 目录 / 文件 | 主要功能 |
| --- | --- |
| `bash_script/train/` | 训练脚本入口（全量、LoRA、xArm 真机数据训练） |
| `bash_script/eval/` | LIBERO 评测脚本（起 policy server、跑单 suite/四套件） |
| `run_libero_eval_tmux_single_ckpt.sh` | 单 checkpoint 的 tmux 快速评测脚本 |
| `scripts/` | Python 入口脚本（`train.py`、`serve_policy.py`、`compute_norm_stats.py`、`xarm_realworld.py`） |
| `src/openpi/` | 核心实现（模型、policy transform、训练配置） |
| `examples/libero/` | LIBERO 环境与评测侧代码 |
| `examples/xarm7/` | xArm 数据转换与真机相关工具代码 |

## 2. 环境配置
仓库基于 openpi官方仓库配置，注意：
- 操作系统：建议 `Ubuntu 22.04`
- GPU：NVIDIA 显卡（训练/推理均依赖 CUDA）
- 显存参考：
  - 推理：`> 8 GB`
  - LoRA 微调：`> 22.5 GB`
  - 全量微调：`> 70 GB`


克隆仓库时需要同步子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

git submodule update --init --recursive
```

官方使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。安装好 uv 后，执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

说明：`GIT_LFS_SKIP_SMUDGE=1` 是拉取 LeRobot 依赖时需要的参数。



本仓库有以下环境变量：

```bash
# 模型权重缓存/下载目录
export OPENPI_DATA_HOME=/path/to/myopenpi/dataset/ckpt

# LeRobot 数据根目录（LIBERO/xArm 数据都放这里）
export HF_LEROBOT_HOME=/path/to/dataset_root

# uv 缓存目录
export UV_CACHE_DIR=/tmp/uv-cache

# 可选：WANDB离线实验记录
export WANDB_MODE=offline
```

### 单独准备 LIBERO 评测环境（examples/libero/.venv）

注意：仓库中的评测脚本会激活Libero环境（`examples/libero/.venv`），所以你需要额外准备这个环境。


```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

deactivate
```

注意还需要设置PYTHONPATH为 `export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero`

## 3. 数据
### 3.1 LIBERO 数据（LeRobot）

训练配置 `pi05_libero` / `pi05_libero_low_mem_finetune` 默认使用：

- `repo_id=physical-intelligence/libero`

离线本地训练时，通常设置：

```bash
export HF_LEROBOT_HOME=/path/to/your/dataset_root
mkdir -p "$HF_LEROBOT_HOME/physical-intelligence"
ln -sfn "$HF_LEROBOT_HOME/libero" "$HF_LEROBOT_HOME/physical-intelligence/libero"
```

### 3.2 xArm 数据转换为 LeRobot

通过原始目录形如 `demonstration_*`，可直接转换：

```bash
export HF_LEROBOT_HOME=/path/to/lerobot_root

uv run examples/xarm7/convert_xarm7_data_to_lerobot.py \
  --data-dir /path/to/dataset/data1 \
  --repo-id local/xarm7_data1
```

转换后，默认用于配置 `pi05_xarm7_finetune`

## 4. pi0.5代码逻辑梳理
### 4.1 训练流程（`scripts/train.py`）

1. 读取配置  
   入口是 `uv run scripts/train.py <config_name>`。  
   `config_name` 会在 `src/openpi/training/config.py` 里解析成 `TrainConfig`（例如 `pi05_libero`、`pi05_libero_low_mem_finetune`、`pi05_xarm7_finetune`）。

2. 构建数据管线  
   `create_data_loader()` 在 `src/openpi/training/data_loader.py`。  
   核心顺序是：
   `repack_transforms -> data_transforms -> Normalize -> model_transforms`。  
   以 LIBERO 为例，`LeRobotLiberoDataConfig` 把数据字段映射到统一键，再经过 `LiberoInputs` 转成模型输入格式。

3. 初始化模型与权重  
   `init_train_state()` 会先按 config 创建模型，再通过 `weight_loader` 加载 base 权重（通常是 `pi05_base/params`），然后构造优化器状态、EMA 状态、分片策略。

4. 单步训练逻辑  
   `train_step()` 做的事很直接：  
   - `model.compute_loss(...)`  
   - 对可训练参数求梯度  
   - `optax` 更新参数  
   - 可选更新 EMA  

### 4.2 pi0.5 的 loss 在代码里怎么定义
`Pi0.compute_loss()` 里，pi05 走的是 flow matching 形式：

1. 对真实 action `a` 采样噪声 `eps ~ N(0, I)`，采样时间 `t ~ Beta(1.5, 1)`（再缩放到 `(0,1)`）。
2. 构造插值点：`x_t = t * eps + (1 - t) * a`。
3. 监督目标是向量场：`u_t = eps - a`。
4. 模型预测 `v_t`（通过视觉 token + 语言 token + action token 一次前向得到）。
5. 损失是 `MSE(v_t, u_t)`（按 action dim 平均）。

pi05 和 pi0 的关键实现差异在于时间条件注入：  
pi05 分支使用 `time MLP + adaRMS`（`embed_suffix()` 里会产出 `adarms_cond` 并传给 PaliGemma）；而 pi0 分支不走 adaRMS，而是把 `action_tokens` 和 `time embedding` 先拼接，再过 `action_time_mlp_in/out` 做融合，得到 action expert token。

### 4.3 推理流程（policy server / 真机本地）

推理统一从 `create_trained_policy()`开始：

1. 加载 checkpoint（JAX `params` 或 PyTorch `model.safetensors`）。
2. 从 checkpoint 的 `assets` 读取 norm stats（不是从训练配置临时读，目的是保证训练/推理归一化一致）。
3. 组装输入/输出 transform：
   - 输入：`repack -> InjectDefaultPrompt -> data_transforms -> Normalize -> model_transforms`
   - 输出：`model_outputs -> Unnormalize -> data_outputs -> repack_outputs`
4. 调 `policy.infer()` 时进入模型 `sample_actions()`。

`sample_actions()` 在 `src/openpi/models/pi0.py`：  
从噪声 `x_1` 出发，用 Euler 离散积分做逆向流（默认 `num_steps=10`），从 `t=1` 迭代到 `t=0`，得到 `x_0` 作为 action chunk。  
实现上先对 prefix（图像+语言）建 KV cache，再只滚动计算 suffix（action token），减少重复计算。

### 4.4 Full fine-tune vs LoRA 

全量微调和 Lora 用同一套训练代码，不同 config：

1. Full（`pi05_libero`）  
   模型参数全开（默认 trainable），EMA 开启。

2. LoRA（`pi05_libero_low_mem_finetune`）  
   - 模型 variant 换成 `gemma_2b_lora` / `gemma_300m_lora`  
   - `freeze_filter` 冻结非 LoRA 参数，只训练 adapter  
   - `ema_decay=None`（LoRA 配置里默认关掉 EMA）


## 5. 训练复现
本仓库提供基于JAX的一键微调脚本，并且对比了用全量微调和Lora在Libero上的表现。由于pi0.5预处理数据时候对动作action normalize 到了 1% - 99% quantile；因此需要额外计算norm statistics。  

### 5.1 LIBERO 全量微调（JAX）

```bash
bash bash_script/train/train_pi05_jax_8gpu.sh
```

### 5.2 LIBERO LoRA 微调（JAX）

```bash
bash bash_script/train/train_pi05_libero_lora_jax.sh
```

### 5.3 xArm7 微调（JAX）

建议先计算 xArm 数据的 norm stats：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_xarm7_finetune
```

然后训练：

```bash
bash bash_script/train/train_pi05_jax_8gpu_realworld.sh
```

## 6. LIBERO 四套件评测
LIBERO 是一个专为机器人操纵任务设计的终身学习基准环境，包含 Spatial、Goal、Object、Long(10) 四个子集。
### 6.1 串行评测四套件（推荐）

```bash
# 可按需覆盖：
# CKPT_DIR=/path/to/checkpoint
# POLICY_CONFIG=pi05_libero 或 pi05_libero_low_mem_finetune
# GPU_ID=0 PORT=8000 NUM_TRIALS=50
bash bash_script/eval/run_libero_eval_pi05_libero_4suites_serial.sh
```

输出目录默认在：

- `data/libero_eval_pi05_libero_4suites/<timestamp>_g<gpu>_p<port>/`

包含：

- `suite_success_rates.txt`
- `all_fail_episodes.txt`
- 每个 suite 的 `eval_summary.txt` / `fail_episodes.txt` / `videos`

### 6.2 单 checkpoint 快速评测（单 suite）

```bash
bash run_libero_eval_tmux_single_ckpt.sh \
  /path/to/ckpt_step_dir \
  0 \
  8000 \
  pi05_libero
```

## 7. Libero实验结果

| 模型/ckpt | libero_spatial | libero_object | libero_goal | libero_10 | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| official | 0.978 | 0.982 | 0.984 | 0.928 | 0.968 |
| lora29999 | 0.974 | 0.968 | 0.966 | 0.902 | 0.953 |
| fulltune29999 | 0.992 | 0.986 | 0.972 | 0.930 | 0.970 |

说明：

- `fulltune29999` 的四套件平均最高（0.970）。
- `lora29999` 相比全量有一定性能回落，但保持了较高成功率（平均 0.953），在较难任务Libero_object 和 Libero_10上掉点明显。
- `libero_10` 是三组里最难的 suite。

## 8. xArm 真机实验
真机数据采集代码参考[这个repo](https://github.com/dzj441/myxarm)；
xarm真机采用一个第三视角相机和一个主视角相机，均为RealSense D435i
需要提前配置好 xarmip 和 相机 ip，即 network.yaml 和 camera.yaml；机器人配置 采用xarm_simple.yaml。

采数据时需要：
```
python robot_camera.py # enbale 相机
python data_collect.py # 采数据，Ctrl+C 终止，自动保存到指定dir
```

数据demonstration格式如下：
```
  cam_0_rgb_video.mp4
  cam_0_rgb_video.metadata
  cam_1_rgb_video.mp4
  cam_1_rgb_video.metadata
  cam_0_depth.h5                  # 可有可无，当前转换脚本不使用
  cam_1_depth.h5                  # 可有可无，当前转换脚本不使用
  xarm_joint_state.h5             # 不使用
  xarm_gripper_state.h5           # positions: [T], timestamps: [T]
  xarm_cartesian_state.h5         # positions: [T,3], rotations: [T,3], timestamps: [T]
  raw_lang.json                   # 语言instruction
```

在真机环境推理需要额外安装`xarm-python-sdk`:
```python
pip install xarm-python-sdk
```
训练好模型后再推理机器中执行：
```bash
uv run scripts/xarm_realworld.py \
  --config-name pi05_xarm7_finetune \
  --checkpoint-dir /path/to/checkpoint_step_dir \
  --prompt "put the lemon into the red cup." \
  --robot-ip 192.168.1.200 \
  --third-cam-ip 192.168.1.201 \
  --third-cam-port 10005 \
  --wrist-cam-ip 192.168.1.201 \
  --wrist-cam-port 10006
```

脚本直接在本地进程加载 checkpoint，不需要单独启动 policy server。  
输入为两路图像 + 7 维状态，动作输出默认取前 7 维（`xyz + rpy + gripper`）。

## 9. 真机实验结果
### 9.1 真机表现及泛化性实验
我们采样一个中等难度的 pick and place 任务作为真机实验，需要机械臂将柠檬放入红色的陶瓷杯中，由于红色的陶瓷杯口较小，需要机械臂精细对准杯口才能准确放入。我们采样了64条demonstration作为训练集，分别全量微调/Lora微调了5000个step，在评测时进行了20次trial，其中10次柠檬摆放位置是seen in dataset；10次柠檬摆放位置unseen in dataset。使用黑布作为背景。

| 模型/ckpt  | seen成功率 | unseen成功率 | 总计 |
| --- | ---: | ---: | ---: |
| lora4999 | 9/10 | 2/10 | 11/20 |
| fulltune4999 | 10/10 | 4/10 | 14/20 |

可以看到pi05在见过的数据上表现稳定，但是泛化性还有改进空间，如果将柠檬放到pi05没见过的初始位置，容易出现失败。
### 9.2 真机鲁棒性实验
一个进阶真机实验是pi05的鲁棒性测试，撤去黑布，将pi05在白天数据上训练，在夜晚进行推理，探究pi05对背景光照和反光的鲁棒性。这里均采用全量微调。

| 模型/ckpt  | seen成功率 | unseen成功率 | 总计 |
| --- | ---: | ---: | ---: |
| 白天训白天测 | 10/10 | 4/10 | 14/20 |
| 白天训晚上测 | 8/10 | 2/10 | 10/20 |
| 晚上训晚上测 | 10/10 | 5/10 | 15/20 |
| 晚上训白天测 | 5/10 | 0/10 | 5/20 |

在训练与推理处于相同时间段（如都在白天或都在夜晚）时，模型表现较好，20 次 trial 约可成功 15 次，并具备一定泛化能力。跨时间段测试时，性能会下降：白天数据训练、夜晚测试会出现成功率下降；夜晚数据训练、白天测试的下降更明显。一个可能原因是白天光照更强，主视角中机械臂存在反光与轮廓模糊，导致位姿感知不稳定；夜晚数据反光更少，模型在该分布下更稳定，但也更容易对夜晚视觉特征过拟合，从而在白天推理时出现更明显退化。

## 10. TroubleShooting

- 训练报 `norm_stats` 缺失：
  - 先执行 `uv run scripts/compute_norm_stats.py --config-name <your_config>`
- 训练报 base params 缺失：
  - 检查 `JAX_BASE_PARAMS` 是否指向包含 `_METADATA` 的目录
- LIBERO 评测启动失败：
  - 检查 `examples/libero/.venv` 是否已安装依赖
  - 检查 `PORT` 是否冲突、`MUJOCO_GL` 是否需改为`glx` 或 `osmesa`

## 11. 致谢

本仓库基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 扩展，感谢原作者团队开源。感谢复旦大学FVL实验室提供xarm机器人。
