# Piper 真机部署（PI05 双 prompt）

`piper` 分支从 `prompt-ablation` 的 `51a94e697ae4a3a59b6996c484c68a3137c6594a`
创建，保留完整的配置、模型、注意力、处理器和 CABO 实现。根目录
`deploy_piper_vlaa.py` 使用此版本 PI05 的 flow 推理，支持双 prompt 和 prompt 消融配置。
`deploy_piper_wyn.py` 是兼容入口，转发到同一实现；两者的参数和源码选择完全一致。

在 AutoDL 重新训练四个任务，使用 `examples/training/train_piper_autodl.sh`，
路径、参数和模型传回方式见 [四任务训练说明](README_TRAIN_AUTODL.md)。

之前的 `lerobot.processor.core` 导入错误和 `num_vlm_prompt_tokens` 缺失说明部署机混用了
不同版本源码。入口现在优先加载旁边的 `src/`，并核对配置来源及必要字段。不要继续向旧版
`PI05Config` 手动补字段，也不要把旧部署目录的 `policies/` 或 `processor/` 复制进来。

## 1. 获取分支和原 Piper 驱动

首次部署可新建目录，保留原部署目录的修改和相机配置：

```bash
cd /home/drx/DJC
git clone --single-branch --branch piper \
  https://github.com/LianYi233/lerobot060.git lerobot-piper
cd lerobot-piper

# 只复制原部署机上已经使用的完整 Piper 驱动目录。
# 首次复制时目标应不存在；若已存在，先检查内容，不要覆盖自己的修改。
test ! -e src/lerobot/robots/piper && \
  cp -a /home/drx/DJC/lerobot-main/src/lerobot/robots/piper src/lerobot/robots/
```

仓库不跟踪部署机的自定义 Piper 驱动，因而该目录不能从 GitHub 拉取。
它至少需要提供 `lerobot.robots.piper.PiperConfig` 和
`lerobot.robots.piper.piper.Piper`，以及原有的 SDK、本地依赖和相机配置。
部署沿用原脚本的驱动约定：`joint_1.pos` 至 `joint_6.pos` 为弧度，`gripper.pos` 为 0..1；
使用前确认这与训练数据和本地驱动一致。
`piper_deploy_guard.py` 适配本次上传的驱动和 `piper-sdk==0.6.1`，通过部署子类覆盖连接、
发送与退出处理；不会覆盖你本地的驱动文件、相机序列号或 RGB 观测转换。

新目录后续更新：

```bash
cd /home/drx/DJC/lerobot-piper
git pull --ff-only origin piper
```

如果已在此仓库的其他分支，先保存本地改动，再用 `git fetch origin`、
`git switch --track origin/piper` 切换；若本地 `piper` 已存在，用 `git switch piper`。
本地同名部署脚本可能阻止切换，不要用强制 checkout 覆盖它。

## 2. 环境检查

使用兼容本分支的 Python 环境。`pyproject.toml` 声明 Python >=3.12、
torch >=2.7 且 <2.12、transformers >=5.4 且 <5.6。优先复用训练环境中的依赖版本，
并保留原部署环境使用的 `piper_sdk`、`pyrealsense2` 及驱动其他依赖。

在已满足依赖的环境里直接运行即可；入口自动加载此 checkout 的源码：

```bash
python deploy_piper_vlaa.py --check_env
```

应打印此目录下 `configuration_pi05.py` 的路径、Python/torch/transformers 版本、
CUDA 是否可用、Piper 驱动文件是否存在，并以 `ENV_OK` 结束。
此检查会导入配置、模型和处理器，不加载权重、不导入或连接 Piper 驱动；
`ENV_OK` 仅代表 PI05 导入通过，驱动显示 `exists=False` 时仍需完成上一步复制。

若创建独立环境，需要联网安装本分支依赖时，可在激活的新环境中执行：

```bash
uv pip install --python "$(command -v python)" -e '.[pi,intelrealsense]'
```

此命令不安装自定义 Piper 驱动或其 SDK；它们应沿用原部署机验证过的版本。
没有 `uv` 时可用 `python -m pip install -e '.[pi,intelrealsense]'`。
不要在旧环境中盲目覆盖 torch/CUDA。离线部署应提前准备同版本依赖。

## 3. 准备 checkpoint 和数据元信息

复制完整 `checkpoints/006000/pretrained_model/`，至少包括：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
处理器 JSON 的 state_file 引用的所有文件（含归一化统计量）
```

入口要求正式 `flow` 阶段的完整 PI05 权重，并核对已保存的 VLM/action prompt 权重形状。
它拒绝缺失 prompt 权重、旧 quantile/regression 模型及仅含 PEFT adapter 的目录。
保留 checkpoint 的 prompt 数量、输入输出特征和 chunk_size，不重新计算部署归一化统计。

`--dataset_info` 指向**该任务训练数据的 `meta/info.json` 副本**。
例如网球任务对应：

```text
May-pick-and-place/3-move_the_tennis_from_yellow_plate_to_blue_plate/meta/info.json
```

可将其复制为 `/home/drx/Downloads/wyn/tennis_info.json`。不需要复制 data/ 或 videos/。
若数据就在本机，也可用 `--dataset_root /路径/May-pick-and-place/3-move_the_tennis_from_yellow_plate_to_blue_plate`
替代 `--dataset_info`。

离线 tokenizer 必须已在本机缓存。如果 checkpoint 记录的是训练服务器上的绝对路径，
为下列命令补充 `--tokenizer_path /本机实际存在的/paligemma-tokenizer目录`。
`HF_HUB_OFFLINE=1` 不会补齐缺失的文件或修复源码版本。

### 脚本在新目录，却仍加载 lerobot-main

如果 traceback 中脚本位于 `lerobot-piper`，但 `configs/policies.py` 来自
`lerobot-main/src/lerobot/`，说明运行的入口仍加载旧安装路径。
`num_vlm_prompt_tokens`、`training_stage`、`cabo_*` 等字段因此会被旧配置类拒绝；
不要从 checkpoint 的 `config.json` 删除这些字段。

若曾把旧脚本复制或重命名为 `deploy_piper_wyn.py`，首次拉取兼容入口前先备份这个
未跟踪文件，避免 Git 拒绝覆盖。以下命令只移动未被 Git 跟踪的同名文件：

```bash
cd /home/drx/DJC/lerobot-piper
if [ -f deploy_piper_wyn.py ] && ! git ls-files --error-unmatch deploy_piper_wyn.py >/dev/null 2>&1; then
  mv -n deploy_piper_wyn.py "deploy_piper_wyn.py.bak-$(date +%Y%m%d-%H%M%S)"
fi
git pull --ff-only origin piper
python deploy_piper_wyn.py --check_env
```

应显示 `/home/drx/DJC/lerobot-piper/src/lerobot/policies/pi05/configuration_pi05.py`
并输出 `ENV_OK`，随后可继续使用原来的 `python deploy_piper_wyn.py ...` 命令。
该入口会自行选择本仓库源码，无需通过重装依赖来更改导入路径。

## 4. 不连接硬件的完整推理

```bash
HF_HUB_OFFLINE=1 python deploy_piper_vlaa.py \
  --policy_path /home/drx/Downloads/wyn/piper-move-tennis-pi05-dualprompt-cabo2-fp32/checkpoints/006000/pretrained_model \
  --dataset_info /home/drx/Downloads/wyn/tennis_info.json \
  --task "move the tennis from yellow plate to blue plate" \
  --dry_run
```

`DRY_RUN_OK` 表示权重和处理器加载后完成了一次合成观测推理。此模式不导入 Piper 驱动、
不连接 CAN 或相机，也不代表真机任务成功。单独使用 `--dry_run` 时元信息可省略；
提供元信息能同时校验状态/动作名称和模型相机 key。

## 5. 真机执行

先停止其他控制程序，按现场流程确认机械臂已停稳，并执行只读诊断：

```bash
python deploy_piper_wyn.py --check_robot --can_port can0
```

该模式不加载模型、不导入相机驱动，不使能、不发送动作或复位命令。
先读取 `ip -json -details -statistics link show dev can0`，要求接口 UP、
`ERROR-ACTIVE` 和 1000000 bit/s；随后仅接收机械臂状态、关节、夹爪和电机反馈。
输出包括 `arm_status`、`err_code`、各关节电压、使能和故障位。
只读检查允许电机处于失能状态，但拒绝控制器故障、反馈过期及 CAN 异常。
`ROBOT_READ_OK` 仅代表当时的通信和反馈检查通过，不代表负载下可靠或可无人值守运行。

若仍显示 `ERROR-WARNING` / `ERROR-PASSIVE` / `BUS-OFF`，先排查机械臂供电、
CAN 接线与接头、终端配置、USB-CAN 连接。不得靠忽略故障、增加发送队列、自动重连
或修改 `restart-ms` 后直接恢复策略来掩盖故障。`error-warn` / `error-pass` 是累计状态
计数，需比较时间上相邻的输出；`bus-off=0` 不等于总线正常。内核的 UVC 错误属于
相机控制请求，单独这一条不能证明 USB-CAN 掉线。

### 相机取帧超时的独立诊断

`Frame didn't arrive within 5000` 表示 RealSense 在 5000ms 内没有提供帧。
`pipeline.start()` 成功并不保证能收到图像，`ROBOT_READ_OK` 也不检查相机。
退出部署、RealSense Viewer、录制及其他使用相机的进程后，在当前环境运行：

```bash
python piper_camera_check.py --list
python piper_camera_check.py --mode single
python piper_camera_check.py --mode together --idle_seconds 10
```

这些命令只操作 RealSense，不加载模型、不导入 Piper SDK、不打开 CAN。
`--list` 应显示部署需要的两台相机，包含序列号、固件和 USB 类型。
不指定 `--camera_serials` 时会检查所有枚举设备；即使检查通过，也不能证明缺失的相机已连接，
更不能自动确认 `cam_high`/`cam_wrist` 与训练视角对应。要核对部署的精确配置，传入相同的
`--camera_serials '{"cam_high":"实际序列号","cam_wrist":"实际序列号"}'`。
默认测试 color BGR8、640×480、30fps，与原驱动默认配置相同；自定义分辨率时传
`--width`、`--height`。相机 30fps 与动作控制 10Hz 是不同参数。

`single` 每次仅启动一台相机，各读取 60 帧；`together` 同时启动所选相机。
`--idle_seconds 10` 在每台相机首帧通过后暂停取帧 10 秒，再读取 60 帧，用于模拟等待 Enter。
相机失败会打印名称、序列号和原始错误，释放全部已打开的相机并返回非零退出码。
仅在测试通过时打印 `CAMERA_CHECK_OK`；该结果不证明后续长时间运行稳定。

如果单台测试失败，检查对应设备是否被占用、连接线/USB 端口、相机配置及 UVC 日志；
如果单台都通过而同时测试失败，重点排查共享 USB 控制器、供电及带宽。
这些只是排查方向，不能仅凭一次超时确定原因。可附上：

```bash
lsusb -t
sudo journalctl -k -b --since "10 minutes ago" --no-pager \
  | rg -i 'uvc|realsense|usb.*(disconnect|reset|error|fail)'
```

部署适配层现会在所有相机启动后、发送使能命令前验证各自的 color 首帧，打印 `CAMERA_READY`。
后续取帧失败仍会停止本次部署并锁定动作；不会用旧图或空图继续推理，也不会自动复位相机。
这项修改用于提前发现并定位错误，不代表已经修复现场 USB 或相机故障。
若本地只保留完整的 `deploy_piper_wyn.py`，仍可使用原入口，但需同时更新
`piper_deploy_guard.py` 和新增的 `piper_camera_check.py`。

硬件问题排除、通信和相机检查通过且现场已准备好后，才能开始动作测试：

```bash
HF_HUB_OFFLINE=1 python deploy_piper_vlaa.py \
  --policy_path /home/drx/Downloads/wyn/piper-move-tennis-pi05-dualprompt-cabo2-fp32/checkpoints/006000/pretrained_model \
  --dataset_info /home/drx/Downloads/wyn/tennis_info.json \
  --task "move the tennis from yellow plate to blue plate" \
  --motion_speed 20 \
  --num_episodes 5 \
  --steps_per_episode 1000
```

连接时先核对 CAN 和反馈，再启动相机并验证 color 首帧，最后进行初始使能；夹爪以反馈的当前开度使能。
每轮按 Enter 开始，轮次间自行重置任务场景。未实现自动回初始位或成功率判定。
原驱动将 `MOVE J` 速度写死为 100%；部署现在使用 `--motion_speed`（1..100，默认 20）。
这不是动作 Hz，也不是碰撞保护：降低此值会改变关节跟踪和任务表现，比较成功率时应记录
并统一该参数。不要将调低速度当作通信故障已修复。

运行时检查每次 CAN 发送返回值：SDK 0.6.1 上层控制接口只记录发送错误而不抛异常，
适配层在其 `GetCanBus().SendCanMessage` 接口拦截返回值。任一帧失败立即抛出
`PiperFaultError`，锁定后续动作，当前 chunk 中剩余帧也不会继续发送。
后台监测覆盖推理和等待 Enter 的时段，检查内核 CAN 状态、反馈时效、电机失能及
碰撞/过流/欠压等故障，首个故障打印 `PIPER_FAULT` 和最近的 CAN/反馈快照。
反馈检查采用 SDK 各消息组的时间戳，超过 0.5 秒拒绝运行；聚合时间戳不能证明每一个
电机反馈帧都及时更新，Python 线程也不提供硬实时保障。

发生故障、Ctrl-C 或正常退出时，都只尝试一次固件快速急停 `MotionCtrl_1(1, 0, 0)`，
然后关闭相机和通信。不再额外调用原驱动的 `DisableArm(7)` 或夹爪失能/清错命令。
**CAN 入队成功不代表机械臂已收到；断电或通信故障时无法保证停止或保持力矩。**
现场必须确认机械臂状态，并按厂商与实验室流程完成停机/支撑。下次测试前需要人工确认
故障已排除并完成必要的控制器恢复；脚本不会自动清错、重新使能或调用 `ResetPiper()`。
SDK 的恢复命令可能导致失力矩，不能将其直接塞进异常处理后重试。

状态与动作分量按数据元信息各自的 `names` 排序。默认执行频率为训练数据 `fps`，
默认每个 chunk 执行 checkpoint 的 `n_action_steps` 步；可用 `--steps_per_chunk 8`
缩短每次执行段，`--control_hz` 只改变执行频率。等待推理会额外增加 chunk 间的间隔，
脚本在这段时间持续发送当前位置，发送失败或反馈异常同样会中止。每次观测只预测一个完整动作段，使用训练时保存的
前后处理器恢复动作单位，随后应用原脚本关节限位和非有限值检查。默认关闭 compile。

相机名称默认去掉模型 key 的 `observation.images.` 前缀，需要改名时显式传入：

```bash
--camera_map '{"observation.images.wrist":"wrist_camera"}'
--camera_serials '{"wrist_camera":"相机的实际序列号"}'
```

示例只说明参数格式；请按 checkpoint 和驱动的实际相机填写所有视角。
图像默认认为驱动输出 RGB；本次上传的驱动请求 RealSense BGR8，并在 `get_observation()`
转换为 RGB，因此保持默认即可。若换用返回原始 BGR 的驱动，传入 `--camera_color_order bgr`。
不要在未检查驱动输出前重复交换通道。

## 本地代码验证

无需 torch 或机器人 SDK 的回归测试（需要 numpy）：

```bash
uv run --no-project python tests/test_deploy_piper.py
uv run --no-project python tests/test_piper_guard.py
# 使用真实 SDK 的编码器及假 CAN 发送端进一步验证，无硬件连接：
uv run --no-project --with piper-sdk==0.6.1 python tests/test_piper_guard.py
```

覆盖源码加载优先级、旧配置拒绝、训练元信息顺序、相机通道、prompt 权重缺失、
归一化处理器文件、动作段执行、异常传播，以及 CAN 发送失败后中止、故障锁定、
后台故障监测、只读诊断和不失能的退出处理。SDK 测试使用真实协议编码器和假 CAN 发送端，
不构造硬件连接；实际机械臂、相机、负载和 checkpoint 仍需在部署机验证。
