# InteractionState v1 to v2 Migration

Phase 1 的 `interaction-state/v1` 保持为刚体冻结基线格式。Phase 2 的适配器将 v1 数据转换为 v2，再进入材料响应层。

## 字段处理

| v1 内容 | v2 处理 |
| --- | --- |
| `schema_version` | 改为 `interaction-state/v2` |
| `scenario_id` | 保留，并新增 `run_id` |
| 后端和实验配置 | 通过 `conditions_sha256` 绑定 `ExperimentManifest` |
| 分组有效性 | 拆为每个观测量的 `quality` 记录 |
| 不可用软体面积 | 使用 `null + unavailable` |
| 估计压力 | 使用 `estimated` 和明确的 `method` |
| NaN | 转换为 `null`，JSON 中禁止 NaN 和 Infinity |
| 稀疏接触点 | 保留在原始后端产物，不塞入逐帧标准状态 |

迁移器必须保留原始 v1 文件，输出 v2 文件和转换日志。转换日志记录源文件哈希、目标文件哈希、字段缺失和估计字段。

