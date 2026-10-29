# 当前完整训练的监控与后续研究

任务：`kevin1220/cifar-resnet-w32-full-39f172`。从第 10 轮检查点续训，目标 120 轮，Kaggle 服务端最长 7,200 秒。用户已授权对明确失败的训练停止。

## 已验证的观测能力

- 已确认任务在 RUNNING，并用 `kaggle kernels logs ... --follow` 接入实时 SSE 日志；训练 epoch 的 loss、准确率、学习率和耗时能逐轮取得。
- 持续日志保存在本实例 `/workspace/.setup/kaggle-dl/competition-research/full-training-w32/live-training.log`。
- `scripts/watch_training.py` 每分钟记录服务器状态与健康判断，终态时下载模型/指标等输出。当前状态在同目录 `watch/monitor.json`；Git 中 `training-watch.json` 只是带时间戳的快照。
- 这是一项训练的有界监控进程，不是已完成的长期 Codex 服务。实例重启后先检查旧进程、远端任务和文件时间，不能盲目重复启动训练。

## 停止判断

- 非有限 loss/accuracy：明确失败。训练代码本身已对非有限 loss 抛错退出；监控也记录停止建议。
- 第 50 轮之后连续 10 轮开发准确率仍低于 60%：记录严重失败和停止建议。
- 第 100 轮之后、学习率 <=0.02，连续 20 轮无至少 0.1 个百分点改善：先核对/保留最佳检查点，再评估提前结束。
- 单轮分数下降、warmup/高学习率阶段波动、暂时没有日志，都不作为自动停止证据。
- 实测每轮约一分钟，慢于短试跑，因此存在 120 轮前触及两小时时限的风险。每轮都有 best/last 检查点；若超时，需确认输出确已保存并验证，再续训剩余轮次。

SDK 有 `cancel_kernel_session`，但该接口需要真实的 kernel session ID，当前按 notebook 引用读取的 API 尚未给出可验证映射；不能拿 kernel ID 代替，也不通过删除 Notebook 来假装停止。因此，**基于开发分数的无人值守远程取消尚未验证**。监控程序明确区分停止建议和实际发出的取消。当前已生效的自动退出保护是非有限 loss 和服务端时限。

## 排行榜

2026-10-29 查询到的公开榜前列为 0.892、0.888、0.884；这是公开评测集分数，不是本项目的开发集准确率，也不能代替最终私有榜。当前账号提交查询返回无记录，剩余允许次数为 5。提交前重新读取实际额度，不能沿用快照。

## 根据当前证据安排方法

宽度 48 的短程比较没有改善，因此本轮先完成宽度 32 基线。当前训练 loss 下降，开发分数有明显波动；暂不足以认定训练失败。

优先研究权重平均（EMA/SWA）以降低权重轨迹噪声，再考虑 Cutout 或 mixup 的单因素实验。已核对：

- Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*, https://arxiv.org/abs/1803.05407 。论文摘要报告在 CIFAR 等任务上通过权重平均改善泛化；不表示本项目已获得同等收益。
- 本地安装的 PyTorch `torch.optim.swa_utils.AveragedModel` 官方 docstring 说明支持 SWA/EMA，并指出 BatchNorm 统计量应通过 `update_bn` 或正确的 buffer 平均策略处理。后续实现不能只平均参数、忽略 BN 统计。
- Cutout / mixup 的来源与待验证假设见 `literature.md`。隐藏标签不参与调参，确认集仍保留到最终一次性检查。

不在健康运行的任务中途更换优化器、数据划分或方法。基线完成后再用相同数据/预算比较，保留失败和不确定的结果。
