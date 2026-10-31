# 独立备用候选：SE-WRN22-4/r16 + 固定 SAM

当前仅实现独立模型及 CPU 数学/参数/序列化/分支位置/梯度测试。
尚无本候选 trainer、原生 GPU 预检、完整开发 benchmark、复测或隐藏预测结果。
不修改已经冻结的 SAM、ResizeMix、WRN 或正在运行的 SWA/Colab 源码与候选。

## 预登记架构与对照

唯一架构为 `se_wrn22_4_r16_v1`：WRN22-4、dropout0、10类；每个残差支路的
conv2 输出后、与 shortcut 相加前加入一个 SE 门，共9个。各 stage 的通道64/128/256，
门的隐藏宽度4/8/16。门采用空间全局均值、带 bias FC、ReLU、带 bias FC、Sigmoid，
只缩放残差支路，不缩放 identity 或 preactivated projection。
实际 CPU 构造的总参数与 trainable 参数均为 **4,332,654**：原 WRN4,298,970加33,684。
最终只有一个 SE 模型；不以冻结参数规避5M限制。

模型先完整调用冻结 `WideResNet` 的 scratch 构造，再由独立新 block 类接管其同名模块，
最后附加新门。共享 state_dict keys 与同 seed 原 WRN 的初始张量逐项相等。
门在 base 完成后使用 `nn.Linear` 默认初始化，正常消耗后续 CPU RNG；不重新初始化共享层，
不使用 forward hook、实例 class 替换、全局 builder monkeypatch、外部权重或预训练。
导出的 model_config 显式含新 architecture identity，必须由独立新 builder 读取；旧 builder 不支持。

[SE 原论文 §III/V-A](https://arxiv.org/html/1709.01507v4)讨论残差相加前的通道门及 CIFAR/WRN/Cutout实验；
[作者实现](https://github.com/hujie-frank/SENet)采用 r=16，
[具体门结构](https://github.com/hujie-frank/SENet/blob/master/models/SE-ResNet-50.prototxt)包含双映射、Sigmoid 和支路相加。
这里的22-4组合、参数预算、初始化与 SAM/Mixup 配方是本项目新实验，不移用论文准确率。

## 固定训练与资源前提

相对于已冻结 SAM，只有架构变化；未来须使用独立新 trainer/来源审计，不能让冻结 trainer/builder
悄悄读取新模型。训练配置以 `research/configs/wrn22_sam.json` 合并原 DEFAULT_CONFIG 为固定基准：

- Primary seed2026，scratch；官方40k训练/5k开发、split seed2026，160轮，batch128。
- SGD Nesterov momentum0.9、LR0.1、min LR0.0001、warmup5、原160轮 cosine、WD0.0005。
- 非 adaptive SAM rho0.05；每个更新复用同一份已增强/Mixup minibatch，第二遍不重采样。
  第二遍 BN 统计保持原 SAM 规则，扰动/失败恢复及实际成功更新计数须单独验证。
- 原 crop/flip、Cutout16、Mixup alpha0.2/prob0.5、label smoothing0、EMA0.995；
  原 raw/EMA 检查点选择规则不改。不另加 ResizeMix、TA、SWA、BN 校准或新 TTA。
- 不接触原确认集，不读隐藏标签，不使用外部训练数据或 checkpoint。

GPU 启动前必须真实 native BF16 两遍 SAM 预检：门参数有限梯度/实际更新、全局扰动 norm、
第二遍 BN、异常恢复与 CPU中断恢复检查；warmup3后完整计时20步，报告 median/p90、显存及全部来源SHA。
不得将 CPU 测试或论文 FLOPs 称为 GPU 速度验证。
预算门在测量前固定：`1.5 * median_step_seconds * 313 * 160 + 300` 秒，
并同时满足单任务3小时、当日剩余 GPU 时间和 campaign hard stop；预检不通过则不启动正式训练。
不新增付费资源购买；不得与已有同资源 lease 并行。

## 唯一主 policy、双门槛与复测

唯一允许晋级的 policy 预先固定为 `center80_shift20`；plain/flip只作完整报告，不替换主 policy。
在相同 v3 source、输入像素、batch、设备/AMP/Torch 上下文下，分别对以下固定参照计算同一门槛：

1. 原 regularized 可信基线的 flip policy（统一工程门槛）。
2. 已冻结 SAM 主 seed2026 的同一 `center80_shift20` policy（新增 SE 的增量门槛）。

第二参照固定为 `results/research/wrn22_sam_s2026/best.pt` 与该目录的 `benchmark.json`，
由既有 `wrn22_sam_delivery/frozen.json` 对应主决策绑定；不是随着后续结果变化的最佳模型。
该 checkpoint SHA为 `5a9771fe26e5b6bd6253a1ddfb2a3187e2087c3348b1569e3585fb6fa6bf07b2`，
benchmark SHA为 `d647ac6bbc734020767c8c3b99f8b1e7e3db842191780c7c8029db9c0554a211`。
正式验证须重新深审实际文件和完整来源，不能只相信计划中的 SHA 或聚合数值。

对每个参照，须满足：clean delta >=0.001 且 stress delta >=-0.002，
或者 stress delta >=0.005 且 clean delta >=-0.002。两个门槛均通过才继续考虑主候选；
不得只继承 SAM 相对旧基线的优势而把它称为 SE 新收益。
固定 raw/EMA 规则选定的主权重须完成5000原图的配对审计；bootstrap按原图携带全部5个stress view，
固定10,000次重采样，保留原图相关性，报告聚合区间与 discordant 计数。
复用、选模后的开发结果仍是探索证据；不按某个 noise/blur case 调 r、门位置、rho 或 epochs。

通过完整 benchmark/配对后，先冻结新配方、主权重、policy、数据/来源SHA及决定时间，
再启动全新 scratch seed2027 完整160轮复测；不得复用旧 seed2027 权重或把后续 tail 时钟当作整次训练开始。
既有 SAM/ResizeMix 复测优先。新候选没有独立来源/模型构造/交付验证前，不走现有冻结提交入口。

## 失败可能与解释边界

SE 门会改变残差尺度，窄 bottleneck 可能抑制有用通道；与 SAM、Mixup/EMA 叠加尚无本项目结果，
不保证收益可加。池化、小矩阵运算和通道乘法会增加实际 kernel 开销；原论文小 FLOPs 增量不是本机吞吐承诺。
自生成 stress 的改善或退化也不说明隐藏测试集的 corruption 类型或 public score 方向。
[SAM 原论文](https://arxiv.org/abs/2010.01412)、[作者代码](https://github.com/google-research/sam)仅提供方法依据。
