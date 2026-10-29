# 训练、续训与生成提交

规则见 `competition.md`、`course-requirements.md`；工作顺序见 `project-plan.md` 与仓库根目录 `research/charter.md`。

## 环境与输入

开发机使用已有 Python 环境；GPU 运行时先检查自身的 PyTorch/CUDA，不套用 CPU wheel 安装命令。项目可在仓库根目录直接通过 `python -m cifar_project...` 运行，也可用 `python -m pip install -e .` 安装；本地复现时保留实际环境版本。

官方数据根目录应包含：

```text
DATA_ROOT/
  train_labels.csv        # id,filename,label; 50,000 rows
  train_images/          # 50,000 32x32 RGB images
  test.csv               # id,filename; 2,000 rows
  test_images/           # 2,000 32x32 RGB images
  sample_submission.csv  # id,label; official IDs and order
```

Kaggle Notebook 直接附加本比赛数据。Colab/本地则将授权数据放到可访问目录；当前全量下载接口 404 的问题不能通过换用外部训练集解决。

## Notebook

`notebooks/train.ipynb` 内含仅由本项目源码与 baseline 配置构成的压缩包，不含数据、权重或凭据，因此在私有仓库尚未公开时也能独立导入 Kaggle/Colab。

1. 选择运行平台，附加比赛数据或设置 `CIFAR10_DATA_ROOT`。
2. 选择 `RUN_MODE`：默认 `inspect`；`pilot` 使用 CUDA 运行头 5 epoch；`full` 运行完整配置；`smoke` 仅作 CPU 管线验证。
3. 为每个独立实验设置新的 `CIFAR_OUTPUT_DIR`。已有 `last.pt` 时会按相同配置续训。
4. 检查打印出的数据规模、参数量、设备和训练曲线。不要将 smoke/验证准确率当成正式测试准确率。
5. 仅在确认模型和数据后开启 `GENERATE_SUBMISSION`，文件只在本地生成，不自动上传。

`notebooks/baseline_smoke_executed.ipynb` 是已执行的官方 60 图小样本验证，有执行输出与图表。最终交付需增加完整 GPU 实验的执行版本，并从已确认的 Kaggle 结果填写官方分数。

源码变更后执行 `python scripts/build_training_notebook.py` 重建自包含 Notebook；该脚本只打包明确列出的 `.py` 文件和配置，不扫描数据/凭据目录。

## 命令行

以下 `$CIFAR10_DATA_ROOT` 为你自己设置的非敏感数据路径。

```bash
python -m cifar_project.train --config configs/baseline.json \
  --data-root "$CIFAR10_DATA_ROOT" --train-csv train_labels.csv \
  --output-dir outputs/baseline-seed42 --device cuda --stop-after-epoch 5

# Pilot 完成后沿用同一个 120-epoch schedule；不要改配置或数据。
python -m cifar_project.train --config configs/baseline.json \
  --data-root "$CIFAR10_DATA_ROOT" --train-csv train_labels.csv \
  --output-dir outputs/baseline-seed42 --device cuda \
  --resume outputs/baseline-seed42/last.pt

python -m cifar_project.predict --checkpoint outputs/baseline-seed42/best.pt \
  --data-root "$CIFAR10_DATA_ROOT" --output outputs/baseline-seed42/submission.csv \
  --device cuda
```

每个训练目录包含配置/环境、固定划分、逐 epoch 指标、best/last 检查点与摘要。续训核对配置、数据/划分指纹、源码、PyTorch、GPU 和设备类型，并恢复优化器、AMP、随机状态和 DataLoader 状态。跨硬件/软件版本不承诺逐位复现；不兼容时创建新实验，不能悄悄重置优化器冒充续训。

推理按官方 sample 的 ID 顺序输出，拒绝缺失/重复 ID、错误列、非法标签、超参数模型和覆盖已存在的提交。输出旁有 SHA256 与 `submitted_to_kaggle: false` 元数据。

## 开发验证

```bash
python -m unittest discover -s tests -v
python scripts/download_smoke_data.py --output-dir data/smoke
python -m cifar_project.train --config configs/baseline.json \
  --data-root data/smoke --train-csv train_labels.csv \
  --output-dir outputs/smoke --device cpu --stop-after-epoch 1 \
  --limit-train-batches 1 --limit-val-batches 1 --smoke-test
```

下载脚本沿用安全环境中的 Kaggle 认证、按类别固定采样并控制请求频率；遇到认证/服务错误直接报告，已下载的图像可保留供恢复。smoke 检查点默认被推理命令拒绝，只有显式 `--allow-smoke-checkpoint` 才能用于非竞赛管线检查。两行 smoke 预测不符合官方 2,000 行提交要求。

功能测试覆盖：残差模型梯度/参数量、三分区隔离与一次性确认、分层划分、图像内容指纹、输入错误、提交契约、连续训练与断点恢复的权重一致性、显式 CUDA 请求和错误配置。

## 引用

- He et al., *Deep Residual Learning for Image Recognition*, https://arxiv.org/abs/1512.03385 。模型为本项目实现的 CIFAR 适配残差 CNN。
- 课程推荐参考：https://github.com/kuangliu/pytorch-cifar 。未加载该仓库权重，也未直接复制其训练实现。
- PyTorch / torchvision，https://pytorch.org/ 与 https://pytorch.org/vision/ 。

## 有预算的研究循环

```bash
python -m cifar_project.research plan
python -m cifar_project.research run --output-dir outputs/research-round1
```

先阅读 `research/charter.md` 和 `research/queue.json`。正式数据划分为 40k/5k/5k；`seed` 控制训练随机性，`split_seed` 固定数据划分，修改初始化不会自动改变验证集。当前两项 10-epoch 筛选只比较宽度，自动记录运行、资源预留、指标和筛选判断，不自动进入长训或比赛提交。模板/源码/数据/环境漂移、缺失产物与失败状态都会中止比较。

选择最终候选后，由负责人执行一次性确认：

```bash
python -m cifar_project.confirm --checkpoint outputs/selected/best.pt \
  --split outputs/selected/split.json --data-root "$CIFAR10_DATA_ROOT" \
  --campaign-dir outputs/research-campaign --device cuda
```

整个研究周期共享同一个确认凭据目录。确认集不是后续调参来源；实际隐藏测试准确率仍只来自经确认的官方结果。终版若在训练+开发数据上重训，须先固定训练时长与方案，不再用确认集逐 epoch 选择检查点；该终版重训模式将在选型后单独实现和验证。
