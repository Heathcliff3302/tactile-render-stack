# K2 力控与回归实现计划

## 状态

**K2-A：CPU 刚体力控参考链路完成，含稠密场与曲线报告。** 250 个测试全部通过，`mechanics.run_k2` 十三项门槛全部通过。

**尚未标记为 K2 完成。** 仅剩一项交付物：Isaac Sim 接触适配器与有效运行时参数采集。由 `validation.milestone_status` 登记，运行报告里 `milestone_complete = false`、`sub_milestone = "K2-A"`、`outstanding = ["isaac_sim_parity"]`——单次运行通过与里程碑完成是两个字段，不能互相读成对方。

## 力控参数的三道天花板

每一道都是"一个被声明却无人核验的数字"，清掉一道不代表清掉另一道。三道都在加载期校验。

| 天花板 | 条件 | 约束什么 | 检查位置 |
| --- | --- | --- | --- |
| 驱动钳位 | `max_press_speed × mass / output_dt ≥ F_target + tol` | 能不能发出足够大的命令 | `assess_force_reachability` |
| 积分权限 | `force_ki × force_integral_limit ≥ F_target × output_dt / mass` | 稳态下还能不能维持那个命令 | `assess_regulator_authority` |
| 比例稳定性 | `mass × force_kp / output_dt < 1` | 会不会收敛，还是持续振荡 | `assess_proportional_stability` |

前两道由闭式推导得出，都建立在理想驱动律上。第四项检查用实测补上这个缺口：

| 检查 | 方法 | 覆盖 |
| --- | --- | --- |
| 实测力上限 | 在真实内核上按声明压入上限施力 | 阻尼、重力与补偿、子步、输出区间、采样阈值 |

## 第二道：积分权限（K2 前置发现）

K1 发现照搬 Phase 1 的 `max_press_speed_mps = 0.25` 时 1 N 不可达，并新增 `drive_limits.py` 在加载期校验。那条校验只覆盖了**驱动钳位**这一个天花板。

实现 K2 控制器之前，先把 `k2_force_cpu.json` 声明的增益代入本内核的力学律做闭环仿真，发现**第二个天花板**：

```text
稳态时误差归零，比例项随之归零，命令速度必须由积分项独自供给，而积分项被
force_integral_limit_ns 钳住：

    要求  force_ki × force_integral_limit  ≥  F_target × output_dt / mass
    实际  0.1      × 2.0                   =  0.2 m/s
    需要  1.0 N    × (1/60) / 0.05         =  0.3333 m/s
```

继承增益下闭环的不动点（`saturated_force_ceiling_n`）：

```text
F = a(kp·T + ki·I_lim) / (1 + a·kp),   a = m/dt = 3 N per (m/s)
  = 3(0.2×1.0 + 0.1×2.0) / (1 + 0.6)
  = 0.75 N
```

1200 帧中落在 `±0.05 N` 内的帧数为 **0**，`regulate_force` 会在 `regulate_timeout_s = 20 s` 处中止。

两点必须写清楚，否则会导向错误的修复：

- **`force_kp` 不在条件里。** 把不动点解 `F ≥ T` 会把所有比例项消掉。加大 `kp` 只抬高失败时停住的那个数，不会让目标变得可达。
- **0.75 N 出现了两次，是巧合。** Phase 1 的 0.25 m/s 压入上限恰好也对应 0.75 N，但那是 `frame_force_n` 在钳位处的值；这里的 0.75 N 是 PI 不动点，K2 的压入上限是 0.5 m/s，钳位并不是绑定约束。`assess_regulator_authority` 的 `binding_limit` 字段区分这两者。

修复按 K1 既有的推导惯例（`DERIVED_HEADROOM_FACTOR = 1.5`）：

```text
force_ki ≥ 1.5 × F_target × output_dt / (mass × force_integral_limit) = 0.25
```

| 参数组 | 积分权限 | 余量 | 稳态 | 结论 |
| --- | --- | --- | --- | --- |
| `kp=0.2 ki=0.1 I=2.0`（继承） | 0.2 m/s | 0.60 | 0.750 N | 拒绝 |
| `kp=0.2 ki=0.25 I=2.0`（推导） | 0.5 m/s | 1.50 | 1.00000 N | 采用 |

只改 `force_ki` 一个值，`force_integral_limit_ns` 与 `force_kp` 保持 Phase 1 取值不动。

## 第三道：比例稳定性（PI 稳定性扫描发现）

加入控制周期与反馈延迟的参数扫描后发现的。单位命令速度产生的力是 `a = mass / output_dt`，在声明的一步反馈延迟下误差递推的齐次部分是 `-a·kp`，因此收敛条件为：

```text
a × force_kp < 1      即      force_kp < output_dt / mass
```

**可用的比例增益与输出步长成正比。** 这一点容易漏掉，因为 `force_kp` 看起来像一个无量纲的调参旋钮，而且前面刚得出"kp 不约束可达力"的结论——它确实不约束可达力，它约束的是能不能收敛到目标。

声明的 `force_kp = 0.2` 在声明的 60 Hz 下安全，在更高速率下不安全：

| 控制速率 | `a·kp` | 结果 |
| --- | --- | --- |
| 60 Hz（声明值） | 0.600 | 收敛到 `1.0000 N` |
| 120 Hz | 1.200 | 振荡，停在约 `0.50 N` |
| 240 Hz | 2.400 | 振荡，停在约 `0.30 N` |

实测边界恰在 `a·kp = 1`：`0.99` 仍收敛，`0.9999` 不收敛。因此允许值取边界的 `PROPORTIONAL_STABILITY_MARGIN = 0.9`，不贴边界。`force_kp = 0` 的纯积分调节器在该判据下无条件稳定。

这道校验与前两道正交：权限充足但增益不稳定、增益稳定但权限不足，两种组合都存在且都有测试锁住。

## 已完成

- [x] `drive_limits.py` 增加 `assess_regulator_authority` / `enforce_regulator_authority`，与既有的力目标可达性并列，在 `kernel_config_from_spec` 的加载期一起生效；不可达即拒绝并给出所需 `force_ki`。
- [x] `derive_force_ki` 与 `saturated_force_ceiling_n` 两个推导函数，后者有闭式与迭代两条独立路径互验。
- [x] 相位角色表 `PHASE_ROLES`：`regulate_force` 与 `hold` 同为 `hold` 角色，`lateral_slide_force` 与 `lateral_slide` 同为 `slide` 角色。验收层按角色取帧，不再按字面相位名。
- [x] `ForceFeedbackStateMachine`：六段、PI 律、积分限幅、命令限幅、`settle_frames` 锁定、失联与越界解锁、调节超时中止。
- [x] 控制器协议统一为 `observe(time_s, contact_present, normal_force_n)`；轨迹机接受并声明不读。
- [x] `K1Experiment` → `RigidProbeExperiment`（保留旧名别名），按 `controller.mode` 选状态机；`mechanics/run_k2.py` 入口。
- [x] 验收门槛按模式决定必需集：`run_gate_order(mode)` / `gate_order(mode)`，力控模式多三项门槛。
- [x] 阈值台账双向化：力控规格必须声明 `force_pass_fraction_min`，轨迹规格声明它则报错。
- [x] 三组测试：`test_regulator_authority.py`、`test_force_controller.py`、`test_k2_acceptance.py`。

## 第四道检查：实测驱动上限

前三道都是闭式推导。`F = mass × commanded_speed / output_dt` 对本 CPU 参考是精确的，但**只因为它省略的每一项在这里恰好都为零**：零线性阻尼、重力被声明的补偿精确抵消、每控制步一个子步、无求解器衰减。

把 `linear_damping_per_s` 设为内核确实读取的非零值就能看出问题。阻尼在接触求解前按子步缩放速度，到达约束的速度低于命令速度：

| `linear_damping_per_s` | 压入上限 0.5 m/s 实际产生 | 理想律预测 |
| --- | --- | --- |
| 0 | `1.5000 N` | `1.5000 N` |
| 1 | `1.4750 N` | `1.5000 N` |
| 5 | `1.3750 N` | `1.5000 N` |
| 10 | `1.2500 N` | `1.5000 N` |
| 30 | **`0.7500 N`** — 低于 1 N 目标 | `1.5000 N` |

这五个规格原先全部加载通过，静态检查全部报告可达 1.5 N。`linear_damping_per_s` 的裁决恰恰是 `enforced`——内核确实读它——所以这不是夹带了一个不支持的设置，而是一个**受支持但静态检查看不见**的设置。

`virtual_reality/authority_probe.py` 因此改为**问内核**：用真实配置跑真实求解器，按声明的压入上限施力，报告采样层实际报告的法向力。阻尼 30 的规格现在在加载期被拒绝，并给出实测值、理想律预测值和两者之差。

报告显式列出**未覆盖**的内容：`feedback_delay_steps`、`force_kp`、`force_ki`、`max_correct_speed_mps`、`force_integral_limit_ns`。开环上限不是收敛结论，这部分由闭环扫描回答，两者合起来才构成可达性主张。

## 力反馈源

控制器读**采样后**的法向力，即内核已施加 `sampling.force_threshold_n` 过滤的 `pair.normal_force_n`，与层 2 发布、验收门槛评分的是同一个观测量。选这个而不是求解器原始值，是为了让控制器和它的门槛不可能对"力是多少"产生分歧。采样阈值因此位于回路内部：被判为低于阈值的帧对控制器表现为无接触，于是调节器继续压而不是维持。

## 保持时长的验收语义

比例不等于时长：只保留沉降后 1 帧，带内占比仍是 `1.0`。原先的门槛只看占比，于是一次只保持了 `0.0167 s` 的运行可以通过声明为 `1 s` 的保持要求。

现在门槛同时要求时长：

```text
held_frames ≥ ceil(hold_time_s / control_dt_s)
```

沉降帧**计入**保持时长（`settle_frame_counts_as_held = true`）。这不是任选的：控制器在锁定的那一帧就开始累加 `hold_elapsed_s`，所以锁定帧是第一个保持帧。门槛原先从锁定帧**之后**开始计数，比控制器实际保持的窗口少一帧——实测 59 帧对声明的 60 帧。现在两者测的是同一个窗口。

滑动段加了同样的时长要求，理由相同。`hold_time_s` 与 `slide_time_s` 从 `controller.parameters` 读取，不在 `acceptance` 里重复声明。

## 阈值的唯一归属

力控门槛读 `controller.force` 里的 `target_force_n`、`force_tolerance_n`、`settle_frames`，不在 `acceptance` 里重复声明。一个声明两次的容差可以和自己不一致。`acceptance` 只新增一个真正没有归属的阈值 `force_pass_fraction_min`，它就是 Phase 1 的 `min_slide_force_in_tolerance = 0.9`。

## 运行结果

K1 基线先复现：145 测试通过，九项门槛全通过，472 帧，段长度 `30/175/60/60/117/30` 与冻结文档一致，`spec_sha256` 与 `conditions_sha256` 不变。

K2 运行（`python -m mechanics.run_k2`）：

```text
loop_state completed | 797 帧 | 13.283 s 仿真时间
段长度  pre_contact 30 | approach_force 175 | regulate_force 386
        lateral_slide_force 60 | retract 116 | released 30
```

| 门槛 | 结果 |
| --- | --- |
| 七项 K1 共有门槛 | 全部通过 |
| 力目标达到并保持 | 通过。第 326 帧（`t = 8.867 s`）完成 5 帧沉降锁定，评分 60 帧 = `1.0 s`，达到声明保持时长；带内占比 `1.0`（门槛 `0.9`） |
| 滑动段保持力目标 | 通过。60 帧 = `1.0 s`，达到声明滑动时长；带内 60 帧，占比 `1.0` |
| 调节器在声明权限内 | 通过。最大压入命令 `0.327 m/s`（上限 `0.5`）；受评段仅 `regulate_force` 与 `lateral_slide_force`，共 446 步 |
| 时间步变化不改变数据语义 | 通过，子步 1/2/4 |
| 稠密场与稀疏 Cell 一致 | 通过，逐格比对（详见下节） |
| `InteractionState` 完整回放 | 通过，18 个产物哈希校验 |

峰值报告法向力 `0.981 N`。受评段内最大垂向命令为 `-0.082 m/s`——始终为负，调节器全程未发出向上命令，因此不存在分离后重新接触。

### 反证：不加这道校验会发生什么

临时绕过 `enforce_regulator_authority`、把 `force_ki` 改回 Phase 1 的 `0.1` 再跑：

```text
loop_state   aborted
abort_reason force regulation timeout: the target was not reached and held
frames       1405（其中 regulate_force 1200 帧 = 20 s 超时）
final force  0.750000 N
带内帧数     0 / 1200
是否进入滑动 否
```

1405 帧的运行、恰好停在 0.75 N、一帧都没进过容差带、永远到不了滑动段。没有加载期校验，这会表现为"力控门槛未通过"，看不出问题出在规格本身。

## K1 索引的变化

`experiments/index/k1_single_probe_cpu.json` 相对 K1 冻结版本有两类变化，**物理结果未变**。

新增四个产物（14 → 18）：`dense_contact_field.npz`、`dense_contact_field_metadata.json`、`contact_report.html`、`curve_report_index.json`。K1 与 K2 走同一条装配路径，所以稠密场和曲线报告对两者同时生效。门槛随之 9 → 10（新增 `dense_field_matches_sparse_cells`）。

三个既有产物哈希变化：

- `effective_runtime.json`、`acceptance.json`：新增 `controller_mode`、`required_gates`、`run_acceptance_passed` / `milestone` 等与模式无关的字段。
- `environment.json`：按设计记录主机信息与墙钟时间，每次运行都不同。

`spec_sha256`、`conditions_sha256`、帧数（472）、段长度（`30/175/60/60/117/30`）全部不变，十项门槛全部通过。

## PI 稳定性扫描

权限检查只说明"理论上有足够的积分权限"，不说明在不同控制周期、延迟和限幅组合下会收敛。`test_force_stability.py` 因此闭合回路——被控对象是内核自己的力学律，反馈延迟按声明步数显式入队——按三个性质评分：不发散、不持续振荡、在声明超时前进入容差带。

扫描覆盖：控制周期 60/120/240 Hz、反馈延迟 1/2/4 步、`force_kp` 0 到稳定边界的 0.9、`settle_frames` 1/5/20。两条增益都按速率推导而非固定，因为两条的要求都随输出步长变化——把 60 Hz 的声明值写进每一行，正是这个扫描第一次运行时失败的原因，也正是它发现第三道天花板的方式。

扫描包含一个**必须失败**的用例（`force_ki = 40`），用来证明这些断言确实能失败；还有一个用例直接验证实测边界与推导边界一致。声明规格本身单列一项，验证它落在稳定包络内且收敛时间不到声明超时的一半。

## 里程碑状态与单次运行验收分离

`validation.milestone_status` 登记每个模式的里程碑交付物及其状态。一次通过的 CPU 参考运行是**一个后端、一个场景**，无法确立跨后端的里程碑，因此报告里是两个字段：

```text
run_acceptance_passed  true    这一次运行的全部门槛都发出且通过
milestone_complete     false   还缺 curve_report / dense_contact_field / isaac_sim_parity
sub_milestone          K2-A    CPU 参考后端，单场景
```

`run_k2` 的退出码只反映本次运行——那是这条命令执行的事——里程碑状态在摘要里单独报告，不与之混同。

## 稠密触觉场

复刻 Phase 1 Step 6.4 的 13 通道布局，通道名与顺序完全一致，既有消费者无需改动即可读取。K2 运行实测：

```text
dense_contact_field.npz        shape = (797, 64, 64, 13)   float32
时间轴帧数 797 == 控制步数 797
段分布     pre_contact 30 | approach_force 175 | regulate_force 386
           lateral_slide_force 60 | retract 116 | released 30
```

**时间轴取自控制步，不取自 cell 记录。** 稀疏行只在有接触时存在，仅凭它建时间轴会悄悄丢掉 `pre_contact` 与 `released`——而"释放后接触力清零"的证据恰恰在那 30 帧里。门槛显式校验这两段存在。

**与稀疏记录逐格比对**（`dense_field_matches_sparse_cells` 门槛）：

| 检查 | 结果 |
| --- | --- |
| 帧数 == 控制步数 | 通过 |
| 无接触段保留 | 通过 |
| 未映射的稀疏 cell | 0 |
| 稀疏记录里没有的稠密 cell | 0 |
| `patch_id` 不一致 | 0 |
| 单格力最大误差 | `3.0e-8 N`（float32 存储精度，非求解器精度） |

另以 `cells.csv` 独立复算：447 条稀疏行 ↔ 447 个稠密激活格，力之和 `369.430758` 对 `369.430756`，`released` 段力之和恰为 `0.0`。

**一处对 Phase 1 的有意修正。** Phase 1 在曲率或方向无效时写 `0.0`，使"未定义"与"平坦"、"未移动"无法区分——正是本仓库在别处禁止的伪造零。13 通道布局为兼容保持不变，但有效性标志作为独立的 `[T,H,W]` 布尔数组 `curvature_valid` / `direction_valid` 随张量一同存储，消费者可以分辨两者。

元数据写入 `dense_contact_field_metadata.json`：形状与轴顺序、每个通道的单位/语义/是否可求和（`additive` / `intensive` / `label` / `occupancy`）、网格尺寸与原点、时间步、段分布、`conditions_sha256`。读者不必读源码就能解释这个张量。

`mapped_area_m2` 仍标注为占格代理，`physical_contact_area_m2` 为 `null`。**稠密化只改变存储布局，不提高空间精度，不把代理面积变成物理面积。**

## 曲线报告

`contact_report.html`，单文件、离线可开、无外部资源、无脚本。四张图：

1. 探头高度与法向命令速度
2. 法向力与声明的调节带（`target_force_n ± force_tolerance_n`）
3. 切向速度
4. 接触单元数与占格代理面积

用内联 SVG 直接从运行数组生成，**不新增绘图依赖**——仓库依赖仍只有 `jsonschema` 与 `numpy`，报告不构成扩充依赖表的理由，尤其是一个必须能在无显示器机器上生成的报告。

两处诚实性处理：

- **数据缺口断线，不跨接。** 无接触帧没有力，跨过它连线会画出一段运行从未产生的直线，图会断言数据所否认的连续性。因此未定义值结束当前线段，下一个有定义值另起一段。层 5 的 `measured_force_n` 每帧都有定义，是真正连续的，不在此列。
- **调节带画的是门槛评分依据的那条带**，取自 `controller.force` 的声明值，不是重绘的近似。

报告底部重申边界：占格代理面积、`physical_contact_area_m2 = unavailable`、层 5 `modelled_not_measured`、与 Isaac Sim 之间只有行为与守恒门槛。

## 契约变更

`ExperimentManifest v1` 的 `artifacts[].role` 枚举新增 `dense_contact_field` 与 `report` 两个取值。这是**向后兼容的放宽**：既有清单仍然有效，没有字段改变含义，因此版本号不变。按新 schema 写出的清单用旧 schema 校验会被拒绝，这一点记在此处，不靠读者推断。

## 未做

- [ ] Isaac Sim 适配器与有效运行时参数采集。采集完成之前与 Isaac Sim 之间仍只有行为与守恒门槛，没有数值等价结论。**这是 K2 唯一剩余的交付物。**
- [ ] 清理工作区并提交冻结版本。

## 不做

K4 之前不引入 Hertz、压力场、真实接触面积或材料形变。Point → Cell → Patch 仍是代理接触区域，`physical_contact_area_m2` 与 `model_footprint_m2` 继续为 `unavailable` / `null`。提前引入会把 K2 重新绑回 Isaac Sim 或 GPU，而 Phase 2 的全部意义是把层 2 变成一条开放边界。
