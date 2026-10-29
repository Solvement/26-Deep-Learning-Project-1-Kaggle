# 已核实的比赛要求

比赛：https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview

核实日期：2026-10-29。Kaggle CLI 2.2.5 已成功认证，并导出 API 返回的全部 5 个页面：`data-description`、`Description`、`rules`、`abstract`、`foundational-rules`。以下为这些页面的摘要；API 页面清单不能证明后续公告均已取得；用户已补充 3 页课程 PDF，详见 `course-requirements.md`。讨论 API 当前返回 0 条。

## 任务与模型

- 训练 CIFAR-10 图像分类模型：32×32 RGB，10 类。
- 模型必须为含 residual / skip connections 的卷积网络，可自定义残差块、层数、通道、卷积核、跳连和池化。
- 最终模型最多 **5,000,000 个可训练参数**，包含分类头及其他所有可训练组件。实现时按 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 核验并记录结果。
- 从零训练，禁止任何外部预训练权重、预训练特征提取器和外部训练集。
- 可使用公开代码/参考架构，但必须注明来源并遵守许可。

来源：`Description`；`rules` §1、§3–6、§13。

## 训练数据与方法

- 只使用课程允许的 CIFAR-10 训练数据；可从允许的训练集自行划分训练/验证集。
- 官方文件清单已完整核实：50,000 张 `train_images/` 图像、2,000 张 `test_images/` 图像及三个 CSV，共 52,003 个文件。训练标签来自官方 `train_labels.csv`，每类 5,000 张；元数据审计见 `data-audit.json`。全量图像字节尚未在本地下载，Kaggle 训练使用官方挂载数据。
- 允许数据增强、优化器/学习率调度、batch size/epoch 调整、weight decay、dropout 等实验。
- 允许知识蒸馏的前提：教师模型也使用允许的数据从零训练。量化等方法仍须符合其他规则。
- 隐藏评测数据不得用于训练、微调或官方提交流程以外的验证/模型选择；不得人工标注、查找或重建隐藏标签。
- 已读取的比赛专属规则没有指定必须使用 Kaggle GPU，也没有明确列出 Colab/本地 GPU 禁令；补充 PDF 也允许使用 NYU HPC，未限定单一平台；仍应留意后续课程通知。
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

Kaggle `rules` §16 仍为 TBA；用户提供的课程 `project1.pdf` 第 1 页明确比赛于 **2026-11-06 23:59** 结束，报告于 **2026-11-09 23:59 前**上传。原文未注明时区，计划暂按纽约时间推进，待课程确认。注册截止时间与 Google Form 链接仍未取得，不能据此推断报名或成员变更仍开放。

PDF 还明确报告最多 8 页（含图表与引用）、每位队员上传 Brightspace、报告第一页提供公开 GitHub 链接、代码包含执行 Notebook/图表并打印准确率和参数量；详见 `course-requirements.md`。后续公告和账号实际报名/组队状态仍需持续核对。

## 读取与数据清单验证结果

已成功执行：

```bash
source /workspace/.setup/kaggle-dl/activate.sh
kaggle competitions pages --competition cifar-10-res-net-architecture-challenge-dl --content --format json
kaggle competitions files cifar-10-res-net-architecture-challenge-dl --csv
```

首次连续分页触发 HTTP 429，后续改为保留检查点、每次请求间隔 1.3 秒、拒绝重复文件名，最终完成 261 页、52,003 个唯一文件的枚举。三个 CSV 已下载并核对：`train_labels.csv` 为 `id,filename,label`（50,000 行），`test.csv` 为 `id,filename`（2,000 行），sample 为 `id,label`（2,000 行），索引与样例的 ID/顺序一致，引用的图像路径与完整清单完全匹配。CSV 校验和见 `data-audit.json`。全量下载接口仍返回 404，单文件与 Kaggle 比赛挂载可用。

完整页面 JSON 和 Markdown 已保存在本次云实例的 `/workspace/.setup/kaggle-dl/competition-research/`。这些是实例内证据文件，未作为仓库交付依赖，也不保证跨新任务恢复。重新执行上述只读命令可从 Kaggle 获取最新版本；规则变更时应更新本摘要。

已在授权范围完成私有 5-epoch GPU 工程试跑，并启动有预算的正式研究；当前没有上传比赛预测。进展和比较组见 `research/charter.md`（仓库根目录）及实验记录。
