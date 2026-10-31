# 五天探索研究：先 benchmark，再晋级

先完成已有 Colab 的连接和实际 GPU 预检，再运行通过预算检查的候选。
目前测过的 BN 校准和加权平移 TTA 均未通过晋级门槛。

**材料护照：** `mode=quick` · `sources=verified primary papers/author code + measured local JSON` · `status=exploratory`。记录日期：2026-10-31。

研究窗口和预算以 [campaign.json](campaign.json) 为准。**硬停止时间为
2026-11-05 00:00 America/New_York（EST，UTC-05:00；即 05:00 UTC）**。
截止时停止新实验和提交；监督执行的任务也必须受剩余窗口约束。

## 范围、资源与当前状态

- 原始 160-epoch 主研究、已报告结果和原确认集结论保持冻结。所有新增工作均为探索；
  不再评价原 5,000 张 confirmation 图像，不覆盖原检查点或评分记录。
- 仅使用官方允许的训练图像。开发比较固定为 40,000 张训练、5,000 张 development，
  split seed 2026；最终单个 residual CNN 的总参数和可训练参数均不得超过 5,000,000。
- 用户选择**先用已有 Colab**。官方 CLI 0.7.5 已在 Ubuntu WSL 完成最小权限认证，
  只读余额为 0.00 CU；仅授权 Colab 与基本身份信息，Drive/Cloud 未授权。
  认证成功不代表 GPU 已分配，必须用实际会话、设备和 AMP 预检日志确认。
  不充值，不新增其他 GPU 供应商或付费预算。
- [run_local.py](run_local.py) 已提供本地进程监督、持久 GPU lease、超时和预算账本：
  本地并发 1 个任务、**每 UTC 日最多 6 小时、每个任务最多 3 小时**；训练和完整 benchmark
  都计入任务耗时。该本地监督器不等于 Colab 的调度/预算桥接已经完成。
- [benchmark.py](benchmark.py) 已实现 6 个 case、3 个推理 policy 和晋级 gate；当前相关
  扩展回归由 CI 验证。`wrn22_wd1e4` 已在本地启动，`wrn34_3` 配置已准备；
  SWA、SAM、SE 等新训练方法尚未实现或取得自有实验成绩。

Colab 运行时管理使用 [colab_cli.py](colab_cli.py)，通过官方 0.7.5 entrypoint，
在进程内把请求权限限制为已批准的 Colab/基本身份子集。缓存读取与 refresh 均已实测；
原生凭据留在 WSL 用户目录，目录权限 700、文件 600。`--refresh-auth` 可执行一次
同权限的强制刷新和实际 scope 核验；代码不含任何凭据。Windows 不在官方 CLI 支持范围内，
本机采用现有 Ubuntu WSL；复现安装版本见 [requirements/colab.txt](../requirements/colab.txt)。
运行时分配、真实 GPU 和训练吞吐仍须单独预检，余额为零不代表已获得免费 GPU。

原主研究的四个核心文件不因新研究而修改。新实现、配置、日志和产物应单独保存，
每次保留代码/配置/数据分割/检查点/benchmark 的 SHA-256。

## 已测结果：不因小幅波动而提交

下面是最初 v1 的完整 5,000 张 development 本地消融，batch size 128。
`stress` 是五个自生成变换的准确率算术平均，不是隐藏测试准确率。

| BN 状态与 policy | Clean accuracy | Mean stress accuracy | 结论 |
| --- | ---: | ---: | --- |
| 原 BN，`flip` | 96.28% | 77.552% | 研究比较基线 |
| 原 BN，`center80_shift20` | 96.32% | 77.756% | 未通过 gate |
| 仅训练集加权校准 BN，`flip` | 95.38% | 77.248% | 未通过 gate |
| 仅训练集加权校准 BN，`center80_shift20` | 95.42% | 77.332% | 未通过 gate |

原始 `regularized` 记录仍为 development 96.30%。新基线 96.28% 与其相差一张图像，
来自不同 batch 设置下的数值差异；不覆盖旧记录，也不将这一个样本的差异解释为模型退化。
新候选使用同一研究基线、分割、case 定义和 batch 设置进行比较，并记录设备与精度。

证据文件保留在忽略的本地路径
`results/research/baseline.json`、`results/research/bn_calibration.json`。
BN 校准仅使用 40,000 张优化训练图像，使用按样本数加权的动量
`batch_size / (seen_examples + batch_size)`，避免末尾小 batch 被赋予同等权重。

后续自动晋级使用 `official_development_stress_v2`。v2 统一使用 CPU 噪声发生器，
逐 case 核对像素 SHA，并绑定完整训练历史、已选检查点指标、源码、设备和 AMP 精度。
当前基线由 `campaign.json` 指向 `results/research/baseline_v2_verified.json`：
flip 为 clean **96.28%** / stress **77.628%**；加权平移为 **96.32% / 77.756%**，
仍未过 gate。v1 证据作为历史保留，不与 v2 混合用于新晋级。

### Benchmark 定义与晋级条件

六个 case 为 `clean`、高斯噪声 `noise_004`、`boxblur3`、`contrast065`、
`brightness080`、双线性 32→24→32 的 `resize24`。
这些是从允许的 development 图像生成的工程压力代理，**没有加载 CIFAR-C**，
也不能据此诊断隐藏测试分布或解释此前的 development/public 差距。

三个 policy 为 `plain`、中心与水平翻转 logits 平均的 `flip`，以及
`center80_shift20`：80% 中心/翻转结果，加 20% 四个小平移/翻转结果。
同一完整基线下，以下两条满足一条才通过（单位均为百分点）：

1. Clean 提高至少 **0.10**，同时 stress 降低不超过 **0.20**。
2. Stress 提高至少 **0.50**，同时 clean 降低不超过 **0.20**。

这些是预先设置的探索性工程门槛，不是显著性检验。`--limit` smoke 永远不能晋级。
参数/数据分割/检查点来源必须有效；切换设备或精度时保留新旧日志，避免把数值波动当成收益。
通过 gate 仍不保证隐藏测试提升；每次拟提交前都需要完整 benchmark 及产物校验。

## 五个阶段：按证据前进

阶段可以在窗口内提前完成；没有收益时不为满足日程或提交次数而晋级。

| 阶段 | 动作 | 晋级所需证据 |
| --- | --- | --- |
| 1．BN/TTA 与运行时 | 保留上述已完成的否定结果；优先验证已有 Colab 会话和实际 GPU | GPU/AMP 预检、完整基线及不可覆盖的日志 |
| 2．WD 与深宽配置 | 分别试 `wrn22_wd1e4.json` 和 `wrn34_3.json`；保持其余训练协议一致 | 完整 40k/5k benchmark、参数量、耗时和 gate；失败候选不提交 |
| 3．按收益选择扩展 | 先评估低成本 SWA；SAM、SE 仅在实现、资源和预计收益允许时分别验证 | 新实现测试、真实测速、完整 benchmark；不将组合改动称为单因素收益 |
| 4．复测并冻结 | 对晋级 recipe 复测独立训练 seed，固定权重变体、TTA、BN 和训练时长 | 稳定性证据及冻结配置；不重新读取原 confirmation 评分 |
| 5．固定 full refit | 新模型从零使用全部 50k 允许图像重训；按冻结规则做最终推理 | 不报告与训练重叠的 dev 分数；保存单模型检查点、CSV 和完整回执 |

目前可用的本地受监督入口为 `research/run_local.py --run <unique_id> --config <config_path>`。
它调用冻结训练入口并随后执行完整 benchmark，不自动上传 Kaggle。
不要用裸训练命令绕过监督和预算；Colab 路径必须先完成连接及相同的运行/产物检查。
本地实验 ID 和输出路径必须唯一，既有目录、日志和回执不能静默覆盖。

## 方法证据与待验证假设

1. **BN / TTA。** 作者的 [SWA BN 更新代码](https://github.com/timgaripov/swa/blob/master/utils.py)
   使用按 batch 样本数加权的统计更新。[Better Aggregation in Test-Time Augmentation](https://arxiv.org/abs/2011.11156)
   及其[作者代码](https://github.com/divyashan/test-time-augmentation)表明，TTA 即使整体提高准确率，
   也会把部分正确预测变错。当前本地 BN/TTA 结果决定不晋级，不把更多视图视为已有收益。
2. **Weight decay。** [MixUp 作者训练代码](https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py#L35)
   默认使用 `weight_decay=1e-4`；当前候选只把原 recipe 的 `5e-4` 改成 `1e-4`。
   这是待测假设，不是本项目已证明的最佳值。参考算法/参数，独立实现并遵守引用与代码许可要求。
3. **SWA，尚未实现。** [原论文](https://arxiv.org/abs/1803.05407)和[作者示例](https://github.com/timgaripov/swa#examples)
   支持对 SGD 轨迹权重做平均，最终保留单个模型。作者 CIFAR/WRN 示例从 epoch 161 开始 SWA，
   SWA LR 为 0.05。这里拟议的 40–80 epoch tail、LR 0.01/0.05 都必须单独做 40k benchmark，
   再冻结 full refit；平均后的 BN 只用允许的优化训练图像重新估计。
4. **SAM，尚未实现。** [原论文](https://arxiv.org/abs/2010.01412)与[作者代码](https://github.com/google-research/sam)
   的 CIFAR-10 WRN-28-10 参考配置使用 LR 0.1、WD 0.0005、rho 0.05。
   迁移到预算内 WRN-22-4 是新假设；原大模型成绩不移用。每次更新需要两次反向传播，
   预算先按约两倍计算并实测。实现须复用相同增强/MixUp、控制第二遍 BN 统计，
   并在 AMP 失败时恢复扰动；不得把未实现算法写成完成结果。
5. **架构候选。** 优先比较现有代码支持的 WRN-34-3，再按收益实现 SE。
   其余两种作为备选，实际实现后重新核验全部参数、数值稳定性、显存和真实吞吐。

| 结构候选 | 设计/CPU 核算参数量 | 核验依据与实现边界 |
| --- | ---: | --- |
| SE-WRN-22-4，r=16 | 4,332,654 | 每个残差支路末端、相加前加带 bias 双 FC 门；增加 33,684 参数。[SE 论文](https://arxiv.org/abs/1709.01507)、[作者实现](https://github.com/hujie-frank/SENet/blob/master/models/SE-ResNet-50.prototxt)。该精确组合尚未实现/训练。 |
| WRN-34-3 | 4,166,298 | 现有模型 CPU 构造已计数，配置已准备；[WRN 论文](https://arxiv.org/abs/1605.07146)、[作者参数化代码](https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py)。该精确配置没有可移用的论文成绩。 |
| CIFAR ResNeXt-29，4×24d | 4,845,002 | 按作者 CIFAR 模板核算：hidden 96/192/384，输出 256/512/1024。[论文](https://arxiv.org/abs/1611.05431)、[作者代码](https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)。尚未实现；不同 baseWidth/expansion 约定须重新计数。 |
| PyramidNet-110，α=84，basic | 3,904,446 | 作者 PyTorch 类 CPU 构造计数；论文近似写 3.8M。[论文](https://arxiv.org/abs/1610.02915)、[作者代码](https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/PyramidNet.py)。尚未移植；旧 CUDA padding 必须改为匹配 device/dtype，保留其 basic block 设计。 |

所有候选最终都以实际 `model.parameters()` 的总数和 trainable 数复核，包括 BN、捷径、
门控和分类头。文献 CIFAR 实验仅说明方法依据，不能当成当前比赛成绩或提升承诺。
原 `robust_full` 同时改变增强、训练时长、EMA、BN 和 TTA，不能从其结果推断某一个改动的因果作用。

## 提交与证据

Kaggle 每 **UTC 日、整个队伍合计最多 5 次**。提交前读取实际 API 余额；
只有完整 benchmark 通过、推理规则冻结、CSV 文件 SHA 唯一且本地校验通过的候选才可提交。
没有合格候选时不提交，**不凑满次数**，不重复上传相同预测。

继续使用已有受验证上传流程保存意图、官方 submission reference、文件/检查点/源码 SHA 和评分。
新 research run 只有在接入同等元数据和校验后才可上传，不能绕过 uploader 的 run/source 检查。
原确认集、隐藏标签和外部数据都不用于此轮晋级；隐藏图像只进入最终固定推理和官方提交。
数据、模型权重和预测行保持私有；可发布内容仅限经审计的配置、聚合指标、图表和校验和。

每个任务记录实际设备/精度、配置、数据分割、起止时间、预算、训练历史、benchmark、gate 及失败原因。
到停止时间保存现有证据并结束本轮，不以尚未分配的 Colab GPU、尚未运行的算法或未评分的提交宣布完成。
