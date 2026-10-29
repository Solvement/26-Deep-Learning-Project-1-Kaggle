# 比赛要求与证据

比赛链接：https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview

## 当前状态

截至 2026-10-29，尚未成功读取比赛页面。当前云环境访问 `www.kaggle.com` 时代理返回 403；Kaggle CLI 未配置认证。不能把比赛名称中的 CIFAR-10 或 ResNet 当作已确认的数据集、模型或训练限制。

已安装 Kaggle CLI 2.2.5。其 `competitions pages --help` 支持 `--content` 导出页面正文（使用显式 `--competition` 参数，规避 2.2.5 的可选位置参数与子命令解析歧义），但本比赛的页面列表和权限尚未验证。API 不一定覆盖网页全部栏目、公告、讨论或课程附件。

## 访问恢复后核对

| 来源 | 需要记录的要求 | 状态 |
| --- | --- | --- |
| Overview / Description | 任务目标、作业要求、提交材料 | 待读取 |
| Evaluation | 评分指标、公开/私有榜划分、提交样例与字段 | 待读取 |
| Rules | 模型结构、预训练权重、外部数据、集成、代码公开限制 | 待读取 |
| Data | 文件列表、规模、标签映射、训练/验证/测试划分、数据许可 | 待读取 |
| Timeline | 报名、组队、提交截止时间及原始时区 | 待读取 |
| Team / Submission limits | 是否允许两人组队、成员上限、每日提交次数 | 待读取 |
| Discussion / Announcements / 附件 | 主办方补充或修订规则、课程报告要求 | 待读取 |
| Notebook / GPU | 是否限定平台、运行时长、GPU、网络及输出方式 | 待读取 |

每条确认结果应记录：原文要求、来源链接/栏目、读取时间和实现影响。规则改变时在 PR 中解释影响。

## 获取路径

1. 两名成员分别登录自己的 Kaggle 账号，确认比赛可访问、报名并接受规则；邀请成员加入同一个队伍前核对人数上限。GitHub 协作权限和 Kaggle 队伍权限分别管理。
2. 在云环境设置中保存 Kaggle 网络域名，并安全填写个人 `KAGGLE_API_TOKEN`。不要在聊天、Git、脚本或 Notebook 中粘贴令牌。
3. 激活云环境后执行只读请求：

   ```bash
   source /workspace/.setup/kaggle-dl/activate.sh
   kaggle competitions pages --competition cifar-10-res-net-architecture-challenge-dl --content --format json
   kaggle competitions files cifar-10-res-net-architecture-challenge-dl
   ```

4. 逐项检查网页 Rules、Data、Evaluation 和 Timeline；API 导出的页面不能自动证明已经取得“所有内容”。受课程邀请或登录限制的附件，需要有权限的成员提供正文或导出文件，再进行核对。
5. 在规则明确前，不下载大数据、不启动 GPU 训练、不自动提交比赛。

网络放行或凭据配置不保证比赛授权；403、参赛条款未接受、邀请限制和已结束比赛等情况应分别诊断。
