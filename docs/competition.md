# 已核实的比赛要求

比赛：https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview

核实日期：2026-10-29。Kaggle CLI 2.2.5 已成功认证，并导出 API 返回的全部 5 个页面：`data-description`、`Description`、`rules`、`abstract`、`foundational-rules`。以下为这些页面的摘要；API 页面清单不能证明课程附件、讨论或后续公告均已取得。

## 任务与模型

- 训练 CIFAR-10 图像分类模型：32×32 RGB，10 类。
- 模型必须为含 residual / skip connections 的卷积网络，可自定义残差块、层数、通道、卷积核、跳连和池化。
- 最终模型最多 **5,000,000 个可训练参数**，包含分类头及其他所有可训练组件。实现时按 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 核验并记录结果。
- 从零训练，禁止任何外部预训练权重、预训练特征提取器和外部训练集。
- 可使用公开代码/参考架构，但必须注明来源并遵守许可。

来源：`Description`；`rules` §1、§3–6、§13。

## 训练数据与方法

- 只使用课程允许的 CIFAR-10 训练数据；可从允许的训练集自行划分训练/验证集。
- 标准 CIFAR-10 含 50,000 张训练图像，但课程允许的数据获取地址/交付方式仍需确认；已列出的 Kaggle 文件不代表已完整取得训练集。
- 允许数据增强、优化器/学习率调度、batch size/epoch 调整、weight decay、dropout 等实验。
- 允许知识蒸馏的前提：教师模型也使用允许的数据从零训练。量化等方法仍须符合其他规则。
- 隐藏评测数据不得用于训练、微调或官方提交流程以外的验证/模型选择；不得人工标注、查找或重建隐藏标签。
- 已读取的比赛专属规则没有指定必须使用 Kaggle GPU，也没有明确列出 Colab/本地 GPU 禁令；课程附件和后续通知仍需核对，不能把“未列出”当作主办方正式确认。
- 集成模型的参数预算计算方式等未被明确说明的方案，应在采用前向主办方确认。

来源：`data-description`；`Description`；`rules` §3、§5–6、§12。

## 类别与提交

| label | 类别 |
| --- | --- |
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

提交文件名为 `submission.csv`，恰好两列：`id,label`。label 必须是 0–9 的整数，每张测试图像恰好一条预测；ID、顺序和格式应遵循官方 `sample_submission.csv`，不能只根据说明页示例假定实际 ID。

评分指标为分类准确率。隐藏评测集按固定且不重叠的 **50% 公开榜 / 50% 私有榜** 划分；最终排名仅由私有榜准确率决定，并需通过规则审查。**每天最多提交 5 次，按整队计算。**

来源：`data-description`；`rules` §7–8、§10–11。

## 单人先训练与后续组队

- 每队最多 **3 人**，每名成员只能参加一个队伍。
- 课程要求通过官方 Google Form 注册，注册队名须与 Kaggle 队名一致；该表单链接未出现在已取得的页面正文中。
- 目前按用户一人推进训练准备，其他成员后续加入。每位成员须用自己的 Kaggle 身份完成参赛和正式入队；注册截止后的成员变更需课程教师批准。
- Kaggle 基础规则还对队伍合并的截止时间、历史提交次数等设有条件；后续加入时需结合实际队伍状态核对。
- 禁止跨队私下分享竞赛代码/数据、权重、预测和私有解法。公开代码分享也应遵守比赛规则中关于在比赛讨论/Notebook 面向全部参赛者分享的要求。

来源：`rules` §2、§12；`foundational-rules` §5–6。

## 必须交付的材料

GitHub 仓库须包含：模型实现、训练与评估代码、预处理/增强、优化器/学习率/batch size/epoch 等配置、最终参数量、复现预测的步骤，以及外部代码/方法引用。

项目报告须包含：成员与 Kaggle 队名、架构与设计理由、训练方法、实验对比、最终准确率与参数量、发现/局限/经验、引用及仓库链接。具体报告格式与提交渠道以课程通知为准。教师可能要求检查点、训练日志、实现细节或复现实验。

来源：`rules` §13–15。

## 截止时间与未完成核对

`rules` §16 的队伍注册、比赛提交、报告及 GitHub 提交截止时间均为 **To be announced**，需从课程正式渠道获得具体时间和时区；不能据此推断当前仍开放报名或提交。

尚需核实：课程允许的训练数据来源、完整数据清单与实际 CSV 内容、报名表链接、课程通知/讨论及附件、报告模板、具体时间，以及账号的参赛/组队状态。

## 读取与数据清单验证结果

已成功执行：

```bash
source /workspace/.setup/kaggle-dl/activate.sh
kaggle competitions pages --competition cifar-10-res-net-architecture-challenge-dl --content --format json
kaggle competitions files cifar-10-res-net-architecture-challenge-dl --csv
```

首批文件元数据包含 `sample_submission.csv`、`test.csv`、`test_images/test_0000.png` 等。进一步按页查询文件列表时触发 HTTP 429；未完成全量枚举，也未下载或核对 CSV 内容，不声明文件总数或训练集是否包含在比赛文件中。后续应分批记录分页进度、检查文件名是否重复，并遵循服务端限流信息恢复，不连续重试。

完整页面 JSON 和 Markdown 已保存在本次云实例的 `/workspace/.setup/kaggle-dl/competition-research/`。这些是实例内证据文件，未作为仓库交付依赖，也不保证跨新任务恢复。重新执行上述只读命令可从 Kaggle 获取最新版本；规则变更时应更新本摘要。

此次未启动 GPU 训练、下载大数据或提交比赛预测。
