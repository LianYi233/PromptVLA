# 6000步动作网络微调模型：LIBERO-PRO / LIBERO-PLUS

默认模型为标准 LIBERO 391/400 的这一组：

```text
pi05-curriculum-H2E-e30-h70-S0-steps6000-aexpert-r1-libero-Bocc-base-vlm2-P1-D0-E1-T1
```

加载 `/data1/wyn/chkpt/2601-lerobot/PromptVLA/<RUN_NAME>/checkpoints/006000/pretrained_model/`。
使用完整训练后的 action expert、VLM2 和保存的处理器；不替换成原始 base 权重。
这是 **VLM prompt + 完整 action expert 微调**的结果，不属于 prompt-only 参数量设置。

## 已有两个基准代码和资源时

激活可运行当前 PI05 的 Python 环境，进入 `/home/wyn/VLAA/PromptVLA`。
入口默认从 `/home/wyn/VLAA/LIBERO-PRO`、`/home/wyn/VLAA/LIBERO-plus` 读取对应代码；
安装在其他地方时用 `BENCHMARK_ROOT` 指定实际仓库根目录。

终端一，PRO 官方预生成资源中提供的四类（160个任务，默认每任务10次=1600 episodes）：

```bash
GPU_ID=0 CHECKPOINT_STEP=6000 EVAL_SEED=1000 \
BENCHMARK_ROOT=/home/wyn/VLAA/LIBERO-PRO \
PRO_DIMENSIONS=object,swap,lan,task EPISODES_PER_TASK=10 \
bash run_eval_libero_robustness.sh pro all
```

终端二，PLUS 全部七类、四个suite（10030个变体，各1次）：

```bash
env -u PRO_DIMENSIONS \
GPU_ID=1 CHECKPOINT_STEP=6000 EVAL_SEED=1000 \
BENCHMARK_ROOT=/home/wyn/VLAA/LIBERO-plus EPISODES_PER_TASK=1 \
bash run_eval_libero_robustness.sh plus all
```

PRO 五类包括 `object,swap,lan,task,env`。官方 GitHub 提供 Environment 的生成配置，
但截至2026-10-27检查的 HF 文件目录没有预生成的 `libero_*_env` BDDL/init 文件。
如果本机已经另行生成并核对了完整环境资源，把命令中的 `PRO_DIMENSIONS` 改为 `all`，
即200个任务、默认2000 episodes。仅测四类时输出明确标为 `selected scope only`，
不能当作五类全量成绩。脚本不会将缺失的环境任务替换成普通LIBERO或自动跳过。

PRO 快速单次评估可改 `EPISODES_PER_TASK=1`，但需标明每任务只测1次，
不要与10次/任务结果混用。PLUS固定1次/变体，拒绝误设成10次。

先加 `DRY_RUN=true` 可校验模型和资源、显示最终命令，不加载GPU模型。
它仍会写小型协议清单及预检日志。`PROGRESS=true`默认显示单行进度和ETA，
常规日志写文件；最终只显示suite、扰动类别和总成功率，不保存视频。

## 本机尚未安装基准时

下面用于**新建**的基准目录。已有可用代码/资源时直接指定其根目录，不重复覆盖。
两个fork都使用 `libero` 包名；新入口只在自己的Python进程内选择指定fork，
无需卸载标准 `hf-libero`、无需反复 `pip install -e` 切换，也不修改 `~/.libero`。

建议把已能运行PI05的环境克隆一份用于扩展评估，保留原训练环境：

```bash
conda create -n promptvla-eval --clone lerobot
conda activate promptvla-eval
```

保持现有PyTorch/Transformers版本。PLUS要求robosuite 1.4.1，另外需要Wand和scikit-image：

```bash
python -m pip install 'robosuite==1.4.1' Wand scikit-image
# 以下系统依赖仅在未安装且有sudo权限时安装
sudo apt-get install libexpat1 libfontconfig1-dev libmagickwand-dev unzip
```

使用此前已验证可用的GitHub SSH-over-443连接方式下载独立代码：

```bash
git -c core.sshCommand="ssh -p 443" clone --depth 1 \
  git@ssh.github.com:Zxy-MLlab/LIBERO-PRO.git /home/wyn/VLAA/LIBERO-PRO
git -c core.sshCommand="ssh -p 443" clone --depth 1 \
  git@ssh.github.com:sylvestf/LIBERO-plus.git /home/wyn/VLAA/LIBERO-plus
```

PRO使用自己的仓库assets，不用标准LIBERO目录去覆盖其场景资源。
下载预生成BDDL/init到数据盘，再复制到这个**新建**PRO目录：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
hf download zhouxueyang/LIBERO-Pro --repo-type dataset \
  --include 'bddl_files/*' 'init_files/*' \
  --local-dir /data1/datasets/libero-pro

cp -a /data1/datasets/libero-pro/bddl_files/. /home/wyn/VLAA/LIBERO-PRO/libero/libero/bddl_files/
cp -a /data1/datasets/libero-pro/init_files/. /home/wyn/VLAA/LIBERO-PRO/libero/libero/init_files/
```

PLUS需额外约6.4GB的assets包，不能用标准LIBERO assets替代：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --local-dir /data1/datasets/libero-plus-download

unzip -q /data1/datasets/libero-plus-download/assets.zip \
  -d /data1/datasets/libero-plus-extracted
```

压缩包可能含作者机器的多层路径。以下寻找实际assets目录，仅在目标不存在时建立链接：

```bash
python - <<'PY'
from pathlib import Path
root = Path('/data1/datasets/libero-plus-extracted')
matches = list(root.rglob('scenes/libero_tabletop_base_style.xml'))
if len(matches) != 1:
    raise SystemExit(f'Expected one asset root, found {len(matches)}; inspect extraction')
assets = matches[0].parent.parent
target = Path('/home/wyn/VLAA/LIBERO-plus/libero/libero/assets')
if target.exists() or target.is_symlink():
    if target.resolve() != assets.resolve():
        raise SystemExit(f'{target} already exists; inspect it before replacing anything')
else:
    target.symlink_to(assets, target_is_directory=True)
print(f'PLUS assets: {target} -> {assets}')
PY
```

准备完成后回到PromptVLA目录，运行前面的两条评估命令。
无法访问mirror时可把 `HF_ENDPOINT` 改为可访问的HF服务；资源不会由评估脚本临时生成。

## 协议与结果

- 图像224×224，执行 `n_action_steps=10`，batch1，相对动作控制，关闭compile和AMP；
  处理器统计量、动作归一化、模型权重均来自所选checkpoint。
- 仿真最大动作步数 spatial/object/goal/10 为220/280/300/520，采用基准常用设置。
  这与旧标准LIBERO入口的spatial=280、相机360×360不同，记录在协议文件中。
  对照模型也应使用这个相同入口，不能将这些差异归因于课程策略。
- 指令来自实际BDDL的 `language_instruction`。PRO语言/任务扰动可能保留旧文件名；
  PLUS相机/噪声等虚拟文件名的后缀不会作为语言提示传给模型，真实语言扰动仍保留。
- 每个task有从 `EVAL_SEED/suite/task_name` 确定的独立随机流，断点恢复或分suite执行
  不会因为跳过前面的任务而改变后续模型噪声。并不承诺GPU逐位确定性。
- 预检检查fork来源、任务数、BDDL/init文件、分类与动作权重；记录源代码、任务文件哈希、
  assets文件清单（大小/mtime）及依赖版本。它不等价于完整的GPU/MuJoCo运行测试。

默认结果根目录：

```text
/data1/wyn/chkpt/2601-lerobot/PromptVLA/eval/robustness/
  pro/seed1000-ep10-dimsobject-swap-lan-task/
  plus/seed1000-ep1-dimsall/
```

协议在 `protocol-all/manifest.json`；每个模型的结果目录含 `eval_info.json`、
`task_results/`、`resume_summary.json`和 `robustness_summary.json`。
保留原命令可按任务恢复。不同模型、benchmark、suite、seed、episode次数和PRO维度分别保存。
资源或协议变更时要求新的 `BASE_OUTPUT`，防止混合旧结果。

`robustness_summary.json` 中 `complete` 指所选范围全部完成；`full_benchmark`
才表示覆盖该基准的完整范围。PLUS类别按官方分类文件的**任务名**匹配，避免1-based id错位；
总成功率按实际episodes加权，不对不同大小的suite成功率直接平均。

各suite可以分配到不同GPU，例如 `GPU_ID=2 ... plus libero_goal`，独立结果互不覆盖。
本入口没有自动合并不同命令的结果；完整overall可用同一命令 `all` 完成，
或另行按task级结果去重并核对覆盖后汇总。

标准LIBERO 97.75%可以作为起点表现。PRO/PLUS用于观察扰动泛化；
要归因于“从难到易”，还需相同训练步数、可训练范围和曝光比例的随机顺序对照。

官方资料：

- https://github.com/Zxy-MLlab/LIBERO-PRO
- https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/tree/main
- https://github.com/sylvestf/LIBERO-plus
- https://huggingface.co/datasets/Sylvest/LIBERO-plus
