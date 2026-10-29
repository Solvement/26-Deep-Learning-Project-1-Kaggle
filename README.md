# 26-Deep-Learning-Project-1-Kaggle

用于 2–3 人协作的深度学习项目。开发可在各自本地或云端进行，训练平台可选 Kaggle Notebook、Google Colab 或本地 GPU。

比赛：https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview

## 团队如何工作

Issue 定义任务与验收 → 功能分支开发 → PR → 自动 CI 与另一位成员审查 → 合并 main → 实验验证与版本交付。

- [团队研发工作约定](docs/team-workflow.md)：角色、任务、代码审查、训练记录、agent、异常与交付。
- [仓库协作与保护规则](docs/collaboration.md)：本地双人开发、main 保护和自动化的启用状态。
- [agent 指令](AGENTS.md)：任务范围、冲突处理、验证和审批边界。
- [云端开发与 GPU 选择](docs/cloud-development.md)。
- [比赛要求与待核实项](docs/competition.md)。
- [GitHub/Kaggle 访问配置](docs/api-access.md)。

创建 Issue 时使用开发任务或训练实验模板。代码、配置和测试放入 Git；数据、模型和凭据按工作约定保存在受控位置。

## 当前实现与验证

- CIFAR 适配残差 CNN，默认 **1,084,586 可训练参数**，运行时强制 500 万上限，无预训练权重加载路径。
- 官方 CSV/图像读取、固定 40k/5k/5k 划分、训练与精确续训、实验溯源、预测与提交验证。
- 有预算的私有 Kaggle 研究执行器：方法假设 → 固定条件 → 短程比较 → 收集结果 → 下一轮判断；最终确认集在搜索期间不使用。
- 官方小样本 CPU Notebook 与 Tesla T4 工程试跑已完成。正式宽度 32 基线完成 10 epoch，开发验证最佳 80.88%、最近三轮平均 76.12%；确认集未使用，尚无官方测试成绩。宽度 48 的 10-epoch 最佳开发准确率为 78.52%，最近三轮平均 75.07%；本轮未优于宽度 32。已提交宽度 32 从第 10 轮续训至 120 轮的私有任务。结果见 [实验记录](research/experiment-results.json)。
- Python 功能测试与 GitHub Actions；训练/研究用法见 [训练说明](docs/training.md)，交付约束见 [课程要求](docs/course-requirements.md)。

研究协议见 [research/charter.md](research/charter.md)，首轮假设与预算见 [research/queue.json](research/queue.json)。课程要求最终报告不超过 8 页，并在首页提供公开仓库链接；当前仓库保持私有，公开交付按比赛规则另行处理。

目前没有已确认的官方测试成绩，也没有比赛预测提交。验证/小样本准确率不得冒充隐藏测试成绩。
