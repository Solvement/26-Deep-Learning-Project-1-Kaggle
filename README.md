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

## 当前验证范围

仓库提供 CPU 环境训练自测与 GitHub Actions 配置。自测已在全新 Python 3.12 虚拟环境通过，覆盖依赖兼容、训练、留出集评价、模型恢复、图像转换、CSV 和绘图。远程 Actions 结果及服务端保护规则尚未验证/启用，详见团队工作约定。

比赛完整规则与数据权限尚未核实，目前还没有比赛模型、完整训练入口或比赛成绩；环境自测分数不代表比赛成绩。
