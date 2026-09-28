# Layer 2 Interaction

该目录存放 Isaac Sim、自研接触内核和实测数据到 `InteractionState` 的适配器。适配器只负责单位、坐标、方向、聚合和质量标记，不实现材料响应。

后端原始输出存放到对应实验运行目录。标准逐帧输出写入 `tactile_contract` 定义的 JSONL 或 NPZ 容器。

`kernel_backend.py` 是自研内核的适配器。两条规则决定记录的可信度：

- 内核没有的量标为 `unavailable`，不写入伪造的零。
- 每个观测量在整轮运行中只保留一个计算方法，这正是 `validate_run` 与实验清单绑定的对象。`OBSERVABLE_METHODS` 同时供适配器和清单构造使用，两者不会各写一份而互相偏离。

`manifest_observables()` 由同一张表生成清单的观测量定义。
