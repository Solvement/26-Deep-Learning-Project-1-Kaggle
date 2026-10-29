# 方法研究与尝试顺序

本轮已通过 arXiv 原始摘要页核对以下论文的标题与方法。摘要/论文成绩不能直接转化为本项目的预期成绩；不下载论文作者权重。课程同时推荐 https://github.com/kuangliu/pytorch-cifar ，引用关系见 `docs/training.md`。

| 方法与来源 | 研究发现与可迁移性 | 当前决策 |
| --- | --- | --- |
| He et al., ResNet, https://arxiv.org/abs/1512.03385 | 残差学习改善深层网络优化，论文含 CIFAR 分析；本比赛明确要求残差结构 | 实现适配 32×32 的 basic-block 基线，先验证收敛 |
| Zagoruyko & Komodakis, Wide Residual Networks, https://arxiv.org/abs/1605.07146 | 探索深度/宽度与准确率/效率的取舍，含 CIFAR 实验 | 第一轮只将通道宽度 32 改到 48；不声称本模型是论文原版 WRN |
| Loshchilov & Hutter, SGDR, https://arxiv.org/abs/1608.03983 | 余弦学习率和重启改善 anytime 表现 | 基线采用 warmup + 不重启的 cosine；重启需单独实验，不能把当前实现称为完整 SGDR |
| Zhang et al., mixup, https://arxiv.org/abs/1710.09412 | 在允许训练样本与标签之间插值，改善泛化；论文含 CIFAR 实验 | 观察固定基线的泛化差距后再实现/消融；标签混合与 smoothing 的交互需分开验证 |
| DeVries & Taylor, Cutout, https://arxiv.org/abs/1708.04552 | 随机遮挡训练图像局部的正则方法，可与其他增强结合 | 作为轻量增强候选；需要固定遮挡大小/概率并单因素比较 |

优先级由实际瓶颈决定：未收敛先查输入/优化；训练准确率低再比较容量/优化预算；训练高、开发集明显落后再增加合规正则/增强。复杂方法不能替代数据和复现检查。

后续若采用多阶段资源调度，可参考 Hyperband（https://arxiv.org/abs/1603.06560）和 ASHA（https://arxiv.org/abs/1810.05934）。目前仅采用短程筛选后再延长的原则，尚未逐项复现这些算法；此两项为待深入阅读参考，不能视为本轮已实现或已完整核对。
