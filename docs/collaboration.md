# 双人维护与 agent 协作

## 推荐流程

任务/Issue → topic branch → agent 实现并自测 → PR → agent 提供审查意见 → 另一位人类维护者批准 → 最新 CI 通过 → 授权后 squash merge。

agent 的代码审查建议不能替代另一位维护者的 GitHub approval。PR 作者不能批准自己的 PR；不要用两个人共享的令牌模拟独立审批。处理冲突的 agent 需要理解两边改动、保留贡献并重新验证，涉及实验结论或架构取舍时由维护者决定。

每个任务限定文件范围。两名成员/多个 agent 同时工作时，在任务说明里写明负责人、分支和文件边界；共享模块通过小 PR 串行集成。云任务已有隔离，无需额外创建 worktree。

## 需要仓库管理员启用的规则

当前 GitHub API 访问被代理拒绝，以下设置尚未应用，也未确认仓库订阅是否支持私有仓库分支保护。

- 为 `main` 配置 ruleset / branch protection，要求通过 PR 合并，并至少取得 1 位其他维护者的 approval。
- 新提交后撤销过期 approval；要求最近一次可审查推送由其他人批准；要求讨论解决。
- 将 `CPU smoke` 设为 required status check；先让本 PR 的 workflow 实际运行，确认 GitHub 显示的检查名，再选中它。
- 要求分支基于最新 `main`；禁用强制推送和删除。明确管理员 bypass 策略，日常不绕过规则。
- 建议启用 squash merge。若启用 GitHub auto-merge，仍须以有效人工批准和最新 required checks 为前提。
- 两位维护者用户名尚未提供，因此暂不创建 `CODEOWNERS`。获得真实账号并确认仓库访问权后，可以指定审阅责任；配置多人所有者并不意味着自动要求两人都批准。

## 当前已准备的自动化

`.github/workflows/ci.yml` 在 PR、推送 `main` 和手动触发时，安装 CPU 依赖、检查依赖兼容性，并执行实际训练、验证集评分、模型恢复及输出检查。工作流只有 `contents: read` 权限，不接触 Kaggle 凭据，不使用 `pull_request_target` 执行 PR 代码。

检查目标是开发环境可运行。当前仓库尚无比赛训练实现，不能据此声称模型满足比赛要求。训练代码加入后应补充数据划分、泄漏检查、模型行为和提交格式验证。

GitHub Actions 的运行结果尚未验证；本地执行通过并不代表远程 CI 已启用。新增 workflow 文件也不会自动启用 main 保护。

## CD 与训练产物交付

本项目交付的是可复现的训练代码、Kaggle Notebook、模型和提交文件，不是需要部署的 Web 服务。建议先采用以下流程，待规则和真实训练入口明确后实现：

1. 合并 PR 后，CI 测试代码，记录提交 SHA。
2. 在 选定的 GPU 环境（Kaggle Notebook / Colab / 本地 GPU） 中固定同一提交与环境，使用授权的数据集运行训练；保留配置、随机种子、数据版本和验证分数。
3. 校验推理与提交文件格式，记录模型/文件校验和。
4. 由维护者确认后，手动触发受 GitHub Environment 审批保护的交付或 Kaggle 提交任务。

尚未创建自动提交或 GPU 调度 workflow：比赛提交格式、规则、每日限额、训练入口和保护环境都未确认。未来添加前需验证这些条件，以及 GitHub Environment 的人工审批能力；不得放置空任务并报告 CD 成功。Kaggle 公开榜反馈不得直接作为反复调参的验证集。

## 两人各自在本地开发

两人可以在各自电脑 clone 同一仓库，使用各自 GitHub 身份和不同的功能分支。无需共用工作目录、账户或凭据，也无需先解决 Codex 云环境网络。

```bash
git clone https://github.com/Solvement/26-Deep-Learning-Project-1-Kaggle.git
cd 26-Deep-Learning-Project-1-Kaggle
git switch -c feat/your-task
# 修改文件并执行相关测试
git add <本次修改的文件>
git commit -m "Describe the change"
git push -u origin feat/your-task
```

然后创建 PR。当前 CI 在创建或更新 PR 时运行，也会在 main 更新时运行；单纯推送功能分支不会自动创建 PR。另一位成员用不同分支开发，通过 GitHub 审阅和合并集成。GPU 训练可以在有硬件的成员电脑、Colab 或 Kaggle 进行。

`AGENTS.md` 是 agent 的工作约定，不是常驻后台服务。要实现无人值守的任务执行，必须另行接入 GitHub 的 agent 集成或有权限的 runner，并明确事件触发、可写分支、运行预算、重试次数和人工批准条件。普通 GitHub Actions 不会自动调用当前聊天中的 agent。

可以逐步自动化：PR 创建后的 CI、基于 CI 日志的 agent 修复建议、维护者授权的修复分支和 PR，以及保护规则满足后的 auto-merge。真实语义冲突、规则不明确、实验设计变化和重复失败应交给维护者。比赛提交和付费训练需要单独的授权与保护；不要将它们绑定为每次推送自动执行。
