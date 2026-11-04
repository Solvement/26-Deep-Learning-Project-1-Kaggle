# GPU 与 Kaggle 配置

在仓库根目录用 PowerShell 检查配置。本文记录 2026-11-04 的本机核对结果，运行状态以本地回执和平台实时查询为准。研究由项目所有者 Kevin Yang 授权，Kaggle 团队为 Saber，工作归入 Issue #7。 [研究约定](../research/README.md) · [预算与停止时间](../research/campaign.json)

## 运行环境

| 位置 | 已核验的配置与用途 |
| --- | --- |
| 本机 Windows | 工作区通常在 `%USERPROFILE%\OneDrive\Desktop\Study\NYU\Deep\Project1`。训练用 `C:\Python314\python.exe`（Python 3.14.3），PyTorch 2.12.0+cu130、torchvision 0.27.0+cu130、CUDA runtime 13.0。实际 SAM 回执记录 NVIDIA GeForce RTX 5060 Laptop GPU、7.96 GiB 和 cuDNN 92000。 |
| 本机 GPU 精度 | RTX 5060 的 CUDA 路径实测原生 BF16，SAM 使用 BF16，无 GradScaler。ResizeMix 等特定路径使用显式 FP16 加 GradScaler。以训练配置和该次 `environment.json` 的实际精度为准。`runtime.py` 在导入 Torch 前隔离进程内可能冲突的 cuDNN DLL 搜索路径。 |
| Colab | 历史 Colab 调度通过本机 Ubuntu WSL 的 `~/.local/share/uv/tools/google-colab-cli/bin/python`，CLI 为 `google-colab-cli==0.7.5`，经 `research/colab_cli.py` 限定在 Colab 和基础身份权限。过去用过 Tesla T4 的原生 FP16。上一次记录的余额为 0 CU，后续云分配遇到 HTTP 503，现有记录不能保证 GPU 可分配。认证留在 WSL 用户目录。 |
| Kaggle Notebook | 项目过去使用过免费 T4，分配、额度和比赛数据挂载须在 Notebook 实际核验。Kaggle 提交 API 的认证和 Notebook 的 GPU 分配是两项独立检查。本机核对时共享云端 lease 不存在。 |
| 开发和 CI | Linux CI 跑 CPU 回归，本机 Torch 训练环境和 Kaggle API 工具环境分离。GPU 上正在训练时，只用现有回执做配置核对。 |

原始数据及校验过的缓存位于 `data/`。本机监督回执、日志、权重、预测和提交记录位于私有 `results/research/`。本机使用该环境完成了 `sam_full_refit_nov04_train_20261104_v1`。SAM 训练与预测的进程树均已退出。20:41 UTC 核验时，本机共享 lease 由 `trivialaugment_full_refit_train_20261104_v1` 持有。读 `local_lease.json` 或 `cloud_lease.json` 确认占用，读对应的 `supervisor.json` 和 `environment.json` 确认实际运行。未知清理状态需要先对照原有回执与进程身份。 [平台开发说明](cloud-development.md) · [Colab 固定版本](../requirements/colab.txt)

## Kaggle API 凭据默认值

本机 `kaggle.exe` 位于 `%USERPROFILE%\.local\bin\kaggle.exe`。安装方式为 `uv tool`，默认工具目录可用 `uv tool dir` 查到。本机的独立解释器位于 `%USERPROFILE%\AppData\Roaming\uv\tools\kaggle\Scripts\python.exe`，已核验 Python 3.12.13、Kaggle CLI 2.2.5、kagglesdk 0.1.38。项目入口在训练环境核验模型。API 解释器依次取显式 `--kaggle-python`、存在的 `%APPDATA%\uv\tools\kaggle\Scripts\python.exe`、当前 Python。提交传输校验 Kaggle 包版本为 2.2.5。

本机当前终端的 `KAGGLE_API_TOKEN`、`KAGGLE_USERNAME`、`KAGGLE_KEY`、`KAGGLE_CONFIG_DIR` 和 `KAGGLE_API_V1_TOKEN_PATH` 均未设置。只检查文件存在性时，`%USERPROFILE%\.kaggle\access_token` 存在，同目录的 `access_token.txt`、`kaggle.json` 和 `credentials.json` 均未发现。配置依赖用户目录中的现有 token 文件。不同账号、进程、WSL、CI 和云代理有各自的认证环境。 [此前云代理访问记录](api-access.md)

按本机安装版代码，认证按以下顺序尝试：

1. **访问 token。** Kaggle Notebook 先检查平台提供的 `KAGGLE_API_V1_TOKEN_PATH`。本机先查进程环境变量 `KAGGLE_API_TOKEN`，其次查 `~/.kaggle/access_token` 和 `~/.kaggle/access_token.txt`。已安装 SDK 支持该变量指向现有 token 文件。
2. **旧版 API key。** 使用 `KAGGLE_USERNAME` 和 `KAGGLE_KEY`，或配置目录下的 `kaggle.json`。环境变量优先于旧版文件中的同名设置。Windows 默认配置目录为 `%USERPROFILE%\.kaggle`，`KAGGLE_CONFIG_DIR` 可改此目录。访问 token 文件与 OAuth 缓存仍各按自身默认位置查找。
3. **OAuth。** `kaggle auth login` 通过浏览器登录，将凭据保存在 `~/.kaggle/credentials.json`。以上方式都未成功时，CLI 仅在命令允许的情况下尝试匿名访问。

需要重新配置时，可在 Kaggle 账号设置生成 API Token，再放入用户目录或安全注入进程环境。凭据文件、令牌值和认证调试输出只留在私有环境。访问私有竞赛还需要该账号的参赛权限。 [Kaggle 账号 API 设置](https://www.kaggle.com/settings/api) · [Kaggle CLI 官方认证说明](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)

## 安全核对与提交入口

```powershell
# 版本和来源，仅输出路径与版本
Get-Command kaggle | Select-Object -ExpandProperty Source
kaggle --version
uv tool dir

# 只检查凭据来源是否存在
Test-Path Env:KAGGLE_API_TOKEN
Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.kaggle\access_token') -PathType Leaf

# 只读核验竞赛账号访问和已有提交
kaggle competitions submissions cifar-10-res-net-architecture-challenge-dl --page-size 5

# 只查询团队今日 UTC 提交额度
& C:\Python314\python.exe -c "from research.submit_candidate import KaggleClient; print(KaggleClient().limits())"
```

分配前先核对本地或云端 lease、预检和实时预算。实际提交走当前候选的审计入口，先离线核验模型来源、固定推理策略和官方行，再查询团队实时额度并检查重复及未知提交。原研究候选入口为 [submit_candidate.py](../research/submit_candidate.py)，专用候选使用各自的类型化入口。访问或提交结果未知时，先核对原意图和官方记录。

本轮本地并发上限 1、每 UTC 日 6 小时、单任务 3 小时。Colab 与 Kaggle Notebook 共享云端并发上限 1，Kaggle 研究额外预算上限为 25 GPU 小时，云端额度和本地用量分开。使用已有资源，付费或新增购买需要项目所有者授权。Kaggle 团队提交上限为每 UTC 日 5 次。研究硬停止时间为 **2026-11-05 00:00 纽约时间（05:00 UTC）**。个别路线会为交付预留时间，采用更早的运行截止。 [固定规则](../research/campaign.json)
