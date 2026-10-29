# 云端开发与可切换的 GPU 训练

## 分工

- Codex 云环境：编写代码、运行 CPU 检查、处理 Git 分支和 PR。
- GitHub Actions：每个 PR 的 CPU 自动检查。
- 训练：比赛规则允许后，从 Kaggle Notebook GPU、Google Colab 或本地 GPU 中选择，进行真实训练、推理和产物导出。

当前开发机没有 GPU，安装的是 CPU PyTorch。Kaggle 和 Colab 的 CUDA 运行时由平台提供；本地 GPU 需要匹配硬件、驱动和操作系统。三者都不能直接套用这里的 CPU 安装命令，也不要不加检查地覆盖平台预装的 PyTorch/CUDA。

## 当前云环境

```bash
cd /workspace/26-Deep-Learning-Project-1-Kaggle
source /workspace/.setup/kaggle-dl/activate.sh
python -m pip check
python scripts/smoke_environment.py --output-dir /tmp/kaggle-dl-smoke
```

依赖在 `/workspace/.venvs/kaggle-dl`，安装脚本和可写配置在 `/workspace/.setup/kaggle-dl` 及 `/workspace/.cache`。安装与启动配置已保存到云环境草稿；发布及新任务恢复需要单独验证。

其他 CPU 机器可创建 Python 3.12 虚拟环境后安装：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.0+cpu' 'torchvision==0.24.0+cpu'
python -m pip install -r requirements/ci.txt
python -m pip check
python scripts/smoke_environment.py --output-dir /tmp/kaggle-dl-smoke
```

直接依赖版本已固定；`requirements/ci.txt` 不是完整传递依赖锁。当前云环境的安装脚本另外记录了已验证的完整安装版本。实际比赛训练应另外记录选定 GPU 环境的完整依赖和硬件信息。

## Kaggle Notebook 准备步骤

1. 读取并核对 `docs/competition.md` 的所有未知项；两位成员使用各自账号加入允许的同一队伍。
2. 创建 Notebook 并附加该比赛数据；按账号和比赛允许的范围选择 GPU。检查实际 GPU 是否可用、配额和网络策略，不假定免费 GPU 始终可用。
3. 将已审阅的代码以固定提交导入。若仓库私有，需单独验证认证和 Notebook 网络权限；凭据使用 Kaggle Secrets 或受支持的安全认证，不得硬编码。
4. 先核对数据目录、标签和样例提交；明确数据划分和随机种子，再实现符合规则的 ResNet 训练入口。
5. 先用少量数据验证前向/反向、推理与提交格式，再启动长训练。长训练需要具体任务授权。
6. 保存配置、代码 SHA、依赖版本、数据版本、验证指标、运行日志和模型校验和；检查 Notebook 和输出的可见性是否符合比赛规则。

现已有训练/推理代码、可独立导入的 Notebook 和完成的 GPU 工程试跑；实际记录见 research/charter.md 及 docs/training.md。CPU smoke 仍仅用于工程验证。

## 平台选择与性能比较

| 平台 | 优点 | 需要确认 |
| --- | --- | --- |
| Kaggle Notebook | 便于附加比赛数据和管理比赛输出 | 账号实际 GPU 配额、分配型号、会话限制、网络限制 |
| Google Colab | 云端交互训练，容易切换运行时 | 免费配额和 GPU 不保证；付费选项、会话限制及数据搬运 |
| 本地 GPU | 资源可控、数据与检查点持续保留 | 显存、驱动/CUDA、设备占用、电费和远程访问 |

Kaggle 和 Colab 通常提供有配额的免费 GPU 使用；不能承诺具体账号可用、型号或额度，也不要假定必须购买付费方案。实际价格与限制以平台当时显示为准。付费或长时间资源使用必须获得具体授权。

比赛规则确认后，用相同模型、输入尺寸、精度和 batch size 做短时比较，记录 GPU 型号、显存峰值、吞吐量、每轮耗时和预估总训练时间。必要时说明某平台因显存不足而改变的 batch size。不要仅用平台名称判断性能，也不要在未经确认时承诺某个 GPU 型号。

未来训练入口应将数据路径、输出路径、随机种子和设备写成配置项；显式请求 CUDA 时若 CUDA 不可用应报错，避免无意间在 CPU 上运行长训练。Kaggle、Colab 和本地保持相同训练代码，平台相关的数据挂载与认证放在启动层。迁移前检查数据许可是否允许在该平台使用。

Colab 可从审阅过的固定 Git 提交导入代码，以安全方式访问授权数据；先检查现有 `torch` 版本和 `torch.cuda.is_available()`，再决定是否安装 GPU 依赖。检查点应定期保存到经过授权的持久存储。本地环境同样先确认 GPU 型号、驱动和 CUDA 是否兼容，再按 PyTorch 官方方案安装。
