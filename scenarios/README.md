# Experiment Scenarios

该目录存放长期复用的实验场景和参数清单。每个场景必须包含唯一 ID、版本、输入轨迹、材料、探头、时间基准和观测量定义。

`k0/` 保存从 Phase 1 Step 6 源代码提取的参数清单，以及用于 K1 单探头刚体内核的显式 CPU 参考规格。来源清单中的 `runtime` 参数可能为 `unresolved`，只有完成后端有效值采集后才具备跨后端数值比较资格。

`k1_variants.py` 从 K0 的 CPU 参考派生细化比较用的规格。派生只改数值离散化，并重算 `spec_sha256`，因此比较记录里是一份真实的、可区分的规格，而不是复用同一份规格却声称时间步不同。

## 溯源记录与 CPU 参考

`step6_*.json` 是 Phase 1 源码的**溯源记录**，由 `build_k0.py` 从 Step 6 源码提取并记录其哈希。它们不因本内核的需要而修改——那等于伪造 Phase 1 声明过的值。

`k1_single_probe_cpu.json` 和 `k2_force_cpu.json` 是**显式 CPU 参考**：从对应的溯源记录派生，把 `unresolved` 的运行时换成本内核的设计选择。需要偏离 Phase 1 的参数在这里偏离，并在 `parameter_evidence` 里写明理由。

`k2_force_cpu.json` 的压入上限即是一例：它由力目标推导而非照搬，因为 Phase 1 的 `0.25 m/s` 在本内核上只能到 `0.75 N`，够不到声明的 `1 N`。理由见 [`docs/k1_single_probe_rigid.md`](../docs/k1_single_probe_rigid.md) 的"力的来源与 1 N 目标"。
