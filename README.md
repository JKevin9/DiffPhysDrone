# Vision-based Agile Flight Training Code

## Overview

This repository contains the training code for our research on **Learning Vision-based Agile Flight via Differentiable Physics**. Accepted by [**Nature Machine Intelligence'25**](https://www.nature.com/articles/s42256-025-01048-0). 

[Project webpage](https://henryhuyu.github.io/DiffPhysDrone_Web/) is alive.

## Quick Demos
### Single Agent Flights
<table>
  <tr>
    <td><img src="./gifs/20ms.gif" alt="GIF 1" width="300"></td>
    <td><img src="./gifs/fpv_dense.gif" alt="GIF 2" width="300"></td>
  </tr>
</table>

### Swarm Tasks
<table>
  <tr>
    <td><img src="./gifs/swap_position.gif" alt="GIF 1" width="300"></td>
    <td><img src="./gifs/main_task.gif" alt="GIF 2" width="300"></td>
  </tr>
</table>


## Environment Setup
### Python Environment

The code is tested with the following environment:

- **PyTorch**: 2.2.2
- **Python**: 3.11
- **CUDA**: 11.8

The code should be compatible with other PyTorch and CUDA versions.

### Build CUDA Ops

To build the CUDA operations, run the following command:

```bash
pip install -e src
```

## 当前双 Profile 训练与 Newton Benchmark

本 checkout 已实现 IMPLEMENT_PLAN.md。完整运行命令、输入契约、场景配置和验证边界见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

~~~bash
python main_cuda.py --profile vanilla
python main_cuda.py --profile Newton-Modifed
~~~

默认每种 profile 训练 50,000 iterations；训练不读取 benchmark 地图。用户人工选定新版 checkpoint 后，通过 aerial_lab 的 scripts/competitor/newton/standard_test.py 运行固定测试任务。

## 还原某次训练的代码

每次训练启动时，该 run 的日志目录会写入：

- `git_state.json`：完整 commit、分支、工作区是否有改动，以及 `restore` 命令
- `git_diff.patch`：相对该 commit 的工作区补丁

默认目录是 `logs/<exp_name>/<时间>_<profile>/`。同一份 git 信息会进入 `run_config.json` 和 checkpoint；补丁文件只放在日志目录里。

在一份干净的检出上还原，可以用新的 clone，或 `git worktree add`。这样不会覆盖当前未提交的修改。该 commit 需要已经存在于这份仓库历史中。进入仓库根目录后，按 `git_state.json` 的 `restore` 执行。工作区当时有改动时：

```bash
git checkout <git_state.json 里的 commit>
git apply --binary /path/to/run/git_diff.patch
```

工作区干净时，`restore` 只有 `git checkout`。

补丁还原的是当时磁盘上的文件：已跟踪文件相对该 commit 的全部改动，以及未被 gitignore 的未跟踪文件，二进制文件也在其中。只存在于暂存区、尚未写回磁盘的内容不会进入补丁。场景参数在同目录的 `run_config.json`。gitignore 中的日志与编译产物，以及 conda 环境，需要另行准备。

## Legacy Training Aliases

To start the training process, use the following command:

```bash
# Legacy filename; now selects independent procedural vanilla environments
python main_cuda.py $(cat configs/multi_agent.args)
# Legacy filename; same vanilla profile
python main_cuda.py $(cat configs/single_agent.args)
```

## Upstream Historical Evaluation

The following upstream simulator instructions do not describe the new versioned Newton benchmark checkpoints. See IMPLEMENTATION.md for the current evaluation entry point.

You need to download the simulation validation code from the GitHub release page.
To evaluate the trained model in multi-agent settings, use the following command to launch the simulator:
```bash
cd <path to multi agent code supplementary>
./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json
```

## Citation
If using this repository, please cite our work
```
@article{zhang2025learning,
  title={Learning vision-based agile flight via differentiable physics},
  author={Zhang, Yuang and Hu, Yu and Song, Yunlong and Zou, Danping and Lin, Weiyao},
  journal={Nature Machine Intelligence},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group}
}
```
Then, run the following command to evaluate the trained model:
```bash
python eval.py --resume <path to checkpoint> --target_speed 2.5
```
