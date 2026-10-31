# 五天探索研究：先 benchmark，再晋级

首个 WD 对照已完成 160 轮及完整 benchmark，未通过晋级门槛。
该训练曾中断在第 119 轮，经源码/数据/检查点核验后恢复。中断原因未知，
旧回执及环境记录已保留，该次中断按预留上限保守计入本地预算。
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
  扩展回归由 CI 验证。`wrn22_wd1e4` 已完成并被拒绝晋级，`wrn34_3` 配置已准备；
  SAM 已独立实现并通过 CPU 验证，尚待实际 BF16 测速和正式训练；SWA、SE 尚未实现。

Colab 运行时管理使用 [colab_cli.py](colab_cli.py)，通过官方 0.7.5 entrypoint，
在进程内把请求权限限制为已批准的 Colab/基本身份子集。缓存读取与 refresh 均已实测；
原生凭据留在 WSL 用户目录，目录权限 700、文件 600。`--refresh-auth` 可执行一次
同权限的强制刷新和实际 scope 核验；代码不含任何凭据。Windows 不在官方 CLI 支持范围内，
本机采用现有 Ubuntu WSL；复现安装版本见 [requirements/colab.txt](../requirements/colab.txt)。
运行时分配、真实 GPU 和训练吞吐仍须单独预检，余额为零不代表已获得免费 GPU。

2026-10-31 已实测一台免费 Tesla T4：WRN22-4 为 4,298,970 参数，native FP16、
GradScaler、有限梯度和真实参数更新通过。合成 batch256、dropout0.2、15 步预热和
40 步计时得到 3,760.64 图/秒、中位 68.042ms、P90 68.433ms、峰值 1,024.4MiB。
这不是正式配方的准确率或训练时长估计。预检后 runtime 已停止，active assignments=0，
余额和费率仍为 0；实际缓存过期后的自动 refresh 也通过。公开标量证据见
[research-initial-results.json](../docs/research-initial-results.json)。

原主研究的四个核心文件不因新研究而修改。新实现、配置、日志和产物应单独保存，
每次保留代码/配置/数据分割/检查点/benchmark 的 SHA-256。

最新真实进度：WRN34-3 在本地训练；修正后的 Colab CutMix 任务已通过实际 CUDA
GradScaler 更新/溢出检查并进入真实 40k 训练。T4 的完整 v3 基线为 clean96.28%、
stress77.648%，原生 FP16，六个 case 各5,000张。候选尚未完成，不能报告提分。

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

后续自动晋级使用 `official_development_stress_v3`。v2 统一使用 CPU 噪声发生器，
逐 case 核对像素 SHA，并绑定完整训练历史、已选检查点指标、源码、设备和 AMP 精度。
v2 基线保存于 `results/research/baseline_v2_verified.json`：
flip 为 clean **96.28%** / stress **77.628%**；加权平移为 **96.32% / 77.756%**，
仍未过 gate。v1 证据作为历史保留，不与 v2 混合用于新晋级。

WD1e-4 的 v2 完整结果：flip 为 **96.18% / 77.108%**，加权平移为
**96.32% / 77.236%**；所有 policy 都失败，不生成隐藏预测或消耗提交次数。

v3 明确用 `is_bf16_supported(including_emulation=False)` 检测原生支持，防止 T4 的
模拟 BF16 被误选；训练、case、像素和 gate 不变。当前配置指向独立重测的
`results/research/baseline_v3_verified.json`；所有新候选须与 v3 同设备/精度基线配对。
旧 v1/v2 证据不覆盖、不混用于 v3 晋级。尚未分配的 v2 Colab 输入包已作废，不能启动。

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

### 新增可执行路径：FP16 与 CutMix

[train_shadow.py](train_shadow.py) 是独立的探索训练入口，复用冻结核心函数，
仅允许 40k/5k。T4 使用 native FP16 与 GradScaler；溢出跳步不更新 EMA，
检查点保存 scaler、更新次数和全部 RNG，恢复要求源码/精度/配置相同。
额外训练源码由 [benchmark_shadow.py](benchmark_shadow.py) 核验，随后调用未修改的
v3 benchmark；新候选不能只凭兼容核心的字段绕过独立训练源码检查。

[wrn22_cutmix.json](configs/wrn22_cutmix.json) 实现待测 recipe：WRN22-4、160 epoch、
WD5e-4 不变，用 CutMix alpha1、概率0.5 替换 Mixup 和 Cutout。
[实现](augmentations.py) 按实际裁剪面积重新计算标签权重，供体取自未修改的 batch。
依据为 [CutMix 论文](https://arxiv.org/abs/1905.04899)及[作者代码](https://github.com/clovaai/CutMix-PyTorch)；
论文的大模型/训练时长不同，不能移用其准确率。该比较属于配方替换，不是单因素因果消融。

[run_colab.py](run_colab.py) 默认仅准备私有白名单输入包；显式 `--launch` 才申请一次
现有 Colab T4，不充值。它先测真实 GradScaler 正常更新和溢出跳步，再在同一 T4 上重测
可信原始基线，训练候选并运行完整 benchmark，下载核验产物后停止 runtime。
本地父进程 watchdog 与远端官方 `runtime.unassign()` 截止保护约束最长三小时任务；
不确定清理时保留 cloud lease，禁止盲目再申请。
实际长训练、传输和评分须以回执为准；CPU 测试通过不代表这些云端步骤已完成。

首次正式尝试已分配 T4 并建立远端截止保护，但旧资源检查因 0 CU 余额、1.15 CU/h
而提前停止，未上传数据或启动训练，终止和 active=0 已核实。
CLI 的速率来自 `consumptionRateHourly`，不是美元账单字段；[官方 FAQ](https://research.google.com/colaboratory/faq.html)
说明 CU 用完后回到免费层限制。该组合不证明发生新购买；现有权益可继续使用，
仍禁止充值/订阅/外部 GPU。修正后记录实际余额、速率及单个运行时，不因正 CU/h 自动拒绝。

当前扩展回归含实际训练中断恢复后 model/EMA 完全一致、
CutMix 标签面积、源码/更新计数校验、云端未知分配不重试，以及本地恢复归档失败不覆盖旧回执。

### 小幅收益的配对核验

[paired_compare.py](paired_compare.py) 在相同 v3 环境下复算候选 policy 与原基线 flip，
校验输入像素、准确率、检查点/配置/完整历史及 shadow 来源。然后以 **5,000 个原图 ID**
为抽样单位，每次连同五个 stress 视图共同重采样，固定 seed20261031、10,000 次，
输出 clean/stress 差值的95% percentile区间和配对正确/错误计数。只保存聚合结果，
正确性向量留在进程内存。它不覆盖晋级 gate，也不把复用且已选模的 dev 当成新确认集。

仅在对应 GPU 空闲、已有任务结束并按监督/预算记录后运行；不能与持有 lease 的任务并行。
云端 FP16 benchmark 不能直接与本地 BF16 混用；如改到本地复算，先在同一本地环境
完整重测候选和可信原基线。工具的 CPU 合成检查已通过，尚无新候选的真实配对结论。

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
4. **SAM，已实现、未取得训练结果。** [原论文](https://arxiv.org/abs/2010.01412)与[作者代码](https://github.com/google-research/sam)
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

### 下一轮优化器对照：SAM

[sam.py](sam.py)、[train_sam.py](train_sam.py) 与
[benchmark_sam.py](benchmark_sam.py) 是独立入口，不修改当前 CutMix 或原始训练源码。
配方 [wrn22_sam.json](configs/wrn22_sam.json) 保留原 WRN22-4、160轮、WD5e-4、
Cutout16/Mixup0.2，仅加入非自适应 SAM rho0.05。两次前后向复用同一增强批次，
第二遍仍用批次 BN 归一化，但不重复累计运行统计；精确恢复原权重后只做一次 SGD。
当前只验证 CPU FP32 与本地原生 BF16 路径，FP16/scaler 明确拒绝。

SAM 的21项 CPU检查包含解析二次函数更新、扰动范数、失败恢复、BN只更新一次，
以及实际小模型中断恢复后 model/EMA/SGD/RNG 一致。新增方法后全套108项 CPU检查通过。
这些不构成真实 GPU 性能或比赛准确率。先在空闲 GPU 上运行受监督的
`--benchmark-steps 20` 合成测速，再依据耗时和现有候选结果决定是否投入完整对照。
本地监督器新增 `--trainer sam --precision bf16`，保存训练器/精度并禁止交叉恢复。
云端与本地时间预算分开计数；活动 Colab 任务不会被误计为本地 GPU 时间。

### 新云端对照：ResizeMix 风格变体

CutMix 已完成完整 T4 FP16 v3 benchmark：加权 policy clean96.22%、stress78.896%，
相对于同 T4 基线，stress提高1.248个百分点、clean降低0.06个百分点，符合工程门槛。
它仍缺配对和独立 seed 证据，尚未获准生成提交。产物已取回，原运行时的服务器 assignment
列表和后续用量均核实为0；首次即时清理检查未收敛，原回执及后续核验均保留。

[train_resizemix.py](train_resizemix.py)、[resizemix.py](resizemix.py) 与
[benchmark_resizemix.py](benchmark_resizemix.py) 独立实现下一轮配方，保持WRN22-4/160轮/WD5e-4，
替换旧Mixup/Cutout。整张供体图缩小后粘贴，tau是均匀0.1到0.8的边长比例，标签按真实面积。
每张图独立采样中心，将完整补丁移入边界，不裁掉供体内容。
[作者代码](https://github.com/JieQin-AI/ResizeMix)使用nearest和始终启用；本轮明确采用
**bilinear、概率0.5**，属于待测的受控变体，不宣称复现作者配方或准确率。

云监督器新增 `--trainer resizemix`，仅在实际方法的CUDA几何、FP16正常更新和测速预检通过后
才运行完整基线与训练；原始训练源码和已冻结WRN34交付文件不改。当前配对/交付版本只支持
已有类型；若本方法晋级，须另行提供兼容验证版本，不能绕过源码约束直接提交。

[run_diagnostic.py](run_diagnostic.py) 为测速、独立训练器重测与配对比较提供同一个
本地 lease、预算和 watchdog，不用裸 GPU 命令绕过监督。SAM 实际测速入口为
`python research/run_diagnostic.py --run wrn22_sam_native_preflight --mode sam-speed --config research/configs/wrn22_sam.json`。
只有当前训练结束并释放 lease 后才能运行；输出为新目录中的不可覆盖工程证据。

## 提交与证据

Kaggle 每 **UTC 日、整个队伍合计最多 5 次**。提交前读取实际 API 余额；
只有完整 benchmark 通过、推理规则冻结、CSV 文件 SHA 唯一且本地校验通过的候选才可提交。
没有合格候选时不提交，**不凑满次数**，不重复上传相同预测。

继续使用已有受验证上传流程保存意图、官方 submission reference、文件/检查点/源码 SHA 和评分。
新 research run 只有在接入同等元数据和校验后才可上传，不能绕过 uploader 的 run/source 检查。
原确认集、隐藏标签和外部数据都不用于此轮晋级；隐藏图像只进入最终固定推理和官方提交。
数据、模型权重和预测行保持私有；可发布内容仅限经审计的配置、聚合指标、图表和校验和。

现已为首个合格候选实现独立交付路径：[delivery.py](delivery.py) 先冻结主 seed2026 的
权重/推理 policy 与配对证据，要求选择决定早于独立 seed 复测，复测同一 recipe 和 policy
也须通过完整门槛，才可生成官方2,000行预测。预测使用
`run_diagnostic.py --mode predict` 监督，不占用正在训练的 GPU。
此入口支持有独立开发证据的40k候选；不能把full50k模型伪装成独立验证模型。

[submit_candidate.py](submit_candidate.py) 的 `--prepare` 仅作真实文件校验、不创建上传意图；
`--submit` 重新校验交付链、实际团队配额与 CSV，先持久记录意图再调用官方 CLI/API。
未知上传结果不重传，已知 reference 可用 `--check REF` 只读查询；原五次记录和旧 allowlist 不改。
WRN34 的独立复测、真实2,000行预测和 `--prepare` 已完成，目前等待真实 API 配额。
尚未有新评分；不把 fake API 测试或已生成CSV称为提交成功。

### 按冻结配方全量重训

[run_refit.py](run_refit.py) 为已合格的原始核心 WRN 提供独立50k路径。默认 `--prepare`
只重验已完成的40k交付链、生成不可覆盖的计划与输入配置，不占GPU或预算；显式 `--launch`
才在共享lease/日预算/watchdog内从零训练。只改变 `train_all` 和预先选中的最终权重变体，
其余配方与训练时长继承合格父模型。首次WRN34为160轮。

完整检查50,000训练/0验证、每轮开发指标为null、固定最终轮次/变体、最终权重与last状态完全一致。
[refit_delivery.py](refit_delivery.py) 及 `run_diagnostic.py --mode refit-predict --plan <plan>`
做固定策略推理；上传器只接受经父链和真实full记录深度核验的独立schema。
full模型没有新的独立开发准确率；其依据来自已冻结父配方，不复用重叠dev冒充验证。
该初始全量路径不支持SAM/CutMix/ResizeMix等独立训练器，不能将其强行交给冻结BF16核心。

本地余量由实际账本计算。首次WRN34全量训练估计接近今日SAM后的剩余时间，
计划在2026-11-01 00:00 UTC（10月31日20:00纽约）预算重置后，以至少3,000秒余量门槛启动。
目前只有实现与CPU验证，不能把已准备的计划写成已完成全量训练或已提交。

每个任务记录实际设备/精度、配置、数据分割、起止时间、预算、训练历史、benchmark、gate 及失败原因。
到停止时间保存现有证据并结束本轮，不以尚未分配的 Colab GPU、尚未运行的算法或未评分的提交宣布完成。
