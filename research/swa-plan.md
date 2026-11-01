# 固定40轮SWA tail：预登记协议

登记于2026-10-31，首个SWA GPU预检/训练之前。实际进展与结果见[README.md](README.md)及运行回执；
该协议作为监督任务的哈希输入，在运行中保持不变。

当前优先完成ResizeMix的真实配对/复测与已合格WRN34交付。SWA是后续低增量成本假设，
不因原`robust_full`的多因素失败或某个stress case而挑选学习率。

## 固定配方

- 主seed2026从自身scratch训练的WRN34-3 `last.pt` 第160轮接续，恢复匹配的原始模型、
  SGD momentum及全部RNG。不要把最佳第152轮模型与第160轮optimizer拼接。
- 第161–200轮固定LR **0.01**；保持batch128、WD5e-4、Mixup0.2/p0.5、Cutout16及40k/5k。
  不根据开发结果延长为80轮或改试另一档LR。
- 每个tail epoch末对全部浮点参数做FP32等权在线平均，共40次；包括BN affine参数。
  最终候选固定为这一份平均模型，不按tail development挑选平均窗口/轮次。
- 平均后的BN running buffers单独重建：仅40k训练图像，预登记crop/flip、无Mixup/Cutout，
  按实际batch样本数加权。不得使用5k开发或原确认图像更新统计。
- 同一轨迹的末端raw模型做相同BN更新并保留完整benchmark作对照；它不替代预选SWA候选。
  原父模型、raw-tail、SWA均用同一v3推理上下文记录clean/NLL、五个stress和全部固定policy。

[SWA论文§3.2及附录A.1](https://arxiv.org/abs/1803.05407)的WRN设置使用常数LR0.01；
[作者仓库示例](https://github.com/timgaripov/swa)的默认/示例0.05不同，不能混称完全复现。
[作者BN更新](https://github.com/timgaripov/swa/blob/master/utils.py)支持按实际batch数加权。
这里的模型、数据限制、40轮长度与增强是本项目的新配方，不移用论文准确率。

## 正式复测与晋级

固定40轮tail的增量估计约8–9分钟，另加BN、完整benchmark和来源审计；这是从已测父模型
末40轮耗时外推，尚非本方法GPU测速。若主模型达到原工程gate且原图聚类配对区间支持提升，
本方法预先继承已合格父候选的 **center80_shift20** 作为唯一可晋级policy，完整报告另外两种policy。
除原regularized基线的统一gate外，同一policy还须相对固定WRN34-3主seed2026参照
（clean96.32%/stress78.592%）通过相同阈值gate：clean至少+0.10且stress不低于-0.20个百分点，
或stress至少+0.50且clean不低于-0.20个百分点。该增量门槛在首个SWA GPU预检/训练前固定，
避免只保留父配方的既有优势就将新增训练视为方法收益。两个seed使用同一固定参照。
原v3 `promotion`保留统一基线语义；另记`promotion_vs_parent`，`candidate_eligible`只允许
SWA固定policy同时通过两gate，raw-tail-control始终是对照。随后
冻结配方、主权重和推理policy，再用第二seed **重新从零完成160轮前缀+40轮tail**。
不将此前已看过的seed2027前缀加一段tail的开始时间，当成整个正式重复实验的开始时间。
旧前缀如另做tail只能标为探索，不能代替这一正式复测或进入主隐藏预测。

正式重复的160轮前缀使用原cosine曲线；不能把总epochs改为200后重新计算前160轮LR，
否则不是同一个两阶段配方。新增交付验证须审计父前缀/新增tail的真实更新次数、源码、
配置、时间链、检查点、FP32平均与训练集BN更新。现有v1/v2入口没有SWA支持，不能绕过。

## 上线前的最小验证

验证解析参数平均、BN affine参与平均/统计缓冲重建、原始尾模型与平均模型分离、
SGD/RNG恢复连续性、原生BF16有限更新和实际参数仍为4,166,298。GPU空闲时先做受监督
20步真实更新/测速，再按实际剩余预算运行。单模型参数<=5M；不引入外部数据/预训练。

失败可能包括LR从0.0001升到0.01破坏已有解、高度相关权重平均无收益及BN更新退化。
任何结果均属于复用开发集的探索，不是隐藏分数或原一次性确认结论。

## TrivialAugment排在后面

[TrivialAugment论文](https://arxiv.org/abs/2103.10158)及[作者操作实现](https://github.com/automl/trivialaugment/blob/master/aug_lib.py)
采用逐图独立抽一个操作和均匀强度；标准空间没有Gaussian noise，不能从ResizeMix的noise下降
推导它会补回该项。它需要新的完整训练和逐图增强吞吐验证。既有`robust_full`采用batch共享
RandAugment N2/M9、300轮、EMA0.999、50k、BN重估及10视图，是另一项多因素实验。
