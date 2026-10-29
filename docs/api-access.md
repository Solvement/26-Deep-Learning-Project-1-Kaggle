# GitHub / Kaggle 访问配置说明

## 三层访问彼此独立

1. 应用/插件：某个界面已连接 GitHub 或 Kaggle，并不证明当前云机器能调用同一连接。本任务当前没有可调用的独立 GitHub/Kaggle 插件工具。
2. 网络：云机器的 egress proxy 必须允许目标域名。`Tunnel connection failed: 403 Forbidden` 表示代理在请求到达站点前拒绝连接，不能据此判断账号密码或站点权限错误。
3. 认证和授权：网络接通后，仍需站点有效身份、仓库权限或比赛权限。

GitHub Git 传输和 GitHub REST API 使用不同入口，权限也不能互相推定。

## 本次重新验证的结果（2026-10-29）

- `git ls-remote origin HEAD` 成功。
- `git push --dry-run origin HEAD:refs/heads/chore/cloud-collaboration` 成功；这只验证推送协商，不等于分支内容已经上传，也不保证服务器最终接受 workflow 文件。
- `api.github.com`、`api.kaggle.com`、`www.kaggle.com` 的请求仍被当前实例代理拒绝；提升执行权限后结果相同。
- 配置草稿已包含所需域名，但不能据此声称运行实例已放行。
- `GH_TOKEN` 已注入；不得仅因 API 访问失败而重复索要 GitHub Token。
- `KAGGLE_API_TOKEN` 未注入，个人凭据要求的 `has_saved_binding` 为 false；尚无已保存的 Kaggle 凭据绑定。

## 云端网络

环境配置草稿已经加入以下允许域名，保留原有 package manager 预设：

```text
api.github.com
api.kaggle.com
www.kaggle.com
kaggle.com
```

在环境设置中保存相关配置，并发布更新后的环境，等待产品重新连接后重新验证。保存草稿不会直接修改正在运行的网络代理。不要为了访问一个站点而关闭全部网络限制，也不要绕过代理。

如果用户已完成保存/发布，但新连接仍返回同样的代理错误，需要调查平台配置生效或连接恢复问题，不应反复更换站点 Token。

## Kaggle 认证

登录自己的 Kaggle 账号，在 https://www.kaggle.com/settings/api 生成 API Token。在云环境设置中的个人 `KAGGLE_API_TOKEN` 凭据入口安全填写；其允许目的地是 `api.kaggle.com` 和 `www.kaggle.com`。每位成员使用自己的身份，不把令牌发到聊天或提交到 Git。

随后确认本比赛对该账号可见、已接受参赛规则，且组队条件允许两人协作。API Token 不会自动赋予私有比赛或课程邀请权限。

## GitHub 认证

目前 Git 通道已有平台提供的认证，不需要再提供 Token 来 clone 或测试该仓库的 Git 访问。先解决 API 网络问题，再使用现有 `gh` 注入认证验证仓库 API；只有在站点返回真实的认证/授权错误后，才核对账号、GitHub App 安装的仓库范围和必要权限。

创建/修改 PR、查看 Actions 和配置仓库 ruleset 是不同权限；仓库管理员权限也不等于 agent 应当绕过人工审批。

本地开发时，两人分别使用自己的 GitHub 登录；可以使用 `gh auth login` 或各自配置的 Git HTTPS/SSH 身份。当前云环境网络问题不妨碍各自电脑的正常 Git 操作。
