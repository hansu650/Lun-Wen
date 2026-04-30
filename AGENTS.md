# Workspace Rules

在这个工作区写代码时，默认遵守下面这些规则：

1. 不要写兜底代码。
2. 不要自动猜路径、自动兼容旧版本、自动切换到别的实现。
3. 不要为了“尽量跑起来”加 `try/except` 兜底。
4. 不要额外加一层层 helper、wrapper、fallback 分支。
5. 优先保持原版代码风格：结构简单、入口薄、主线直接。
6. 除非用户明确要求，否则不要额外加显式防御性 `raise` 检查。
7. 能直接写清楚的逻辑就直接写，不要过度抽象。
8. 改动优先最小化，尽量只改当前任务真正需要的部分。
9. 任何没有在当前环境真实跑通、没有明确日志/checkpoint/配置支持的结果，都不能写成当前最好结果。
10. 对于旧 README 中出现但用户已确认无效的信息，应标注为 deprecated，不得作为论文实验结果引用。
11. 当前论文项目的活跃代码目录是 `framework_download/`。
12. 开始实验或整理项目状态前，优先阅读 `README.md`、`docs/experiment_log.md`、`docs/model_changes.md`、`docs/next_steps.md`。
13. 每次实验结束后必须更新 `docs/experiment_log.md`，记录配置、结果、证据路径、结论和下一步。
14. 每次结构改动后必须更新 `docs/model_changes.md`，说明改了什么、为什么改、是否保留。
15. 不要把 deprecated/invalid 记录重新写成有效 baseline，也不要在论文结果中引用。
16. 不要自动引入大型预训练模型，除非用户明确确认当前环境可用。

如果某个子目录里还有更具体的 `AGENTS.md`，则优先遵守那个子目录的规则。
