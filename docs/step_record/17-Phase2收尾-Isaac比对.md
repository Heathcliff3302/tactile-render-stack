---
title: Phase 2 收尾：Isaac Sim 比对（B1–B3）
aliases: [B1, B2, B3, Isaac 比对, isaac_sim_parity]
tags: [触觉渲染, 阶段二, 外部参照, 跨后端, 已完成]
status: 已完成（2026-11-25）
updated: 2026-11-25
---

# Phase 2 收尾：Isaac Sim 比对

## 目标

Phase 2 的本质目的，是跳过 Isaac Sim 做出自己的触觉内核。但在这之前，我们一直只能说"解耦已完成，等价未验证"。这一步用 Isaac Sim 作为参照，回答"我们的内核算得对不对"，并据此把 Phase 2 收尾。

**收尾判据由用户在开工前确定**（2026-11-25），阈值冻结在 `validation/isaac_parity.py` 的 `THRESHOLDS` 中：

| 门槛 | 阈值 |
| --- | --- |
| 接触起始 | 两边相差 ≤ 1 个控制步 |
| 接触点位置 | ≤ 0.1 mm |
| 接触法线 | ≤ 1° |
| 力控稳态 | 两边都进入目标 ±5%，并在末 30 帧保持 |
| 动量平衡 | 两边各自 ≤ 1e-6 N·s |
| 力的瞬态 | 只报告，不作门槛 |

时间盒 2 周，不包含盒形探头，实际一天完成。

## 环境

| 项 | 取值 |
| --- | --- |
| 机器 | AutoDL，RTX PRO 6000 Blackwell Server Edition（96 GB） |
| 驱动 | **580.95.05**。595.71.05 下两次都在 `librtx.scenedb.plugin` 段错误，已确认要避开 |
| Isaac Sim | 5.1.0，pip 安装（没有 Docker），Python 3.11 |
| 一键脚本 | `Phase_2/docs/autodl_isaac_setup.sh` |

环境上踩到的坑：

1. 容器带了 NVIDIA 的 Vulkan ICD，但缺 libX11、libXext，Vulkan 只认到 CPU 的 llvmpipe；缺 libGLU、libXt，MaterialX 和 Iray 加载失败。
2. AutoDL 的学术加速只对 GitHub、HuggingFace 有效，开着它装 pip 包反而超时。NVIDIA 的包直连下载，其他包走阿里云镜像。
3. `import isaacsim` 会弹出 EULA 确认，后台任务必须用 `OMNI_KIT_ACCEPT_EULA=YES` 接受。
4. Isaac 出了异常，进程照样返回 0；`SimulationApp.close()` 会直接结束进程。所以结果要在 `close()` 之前写盘，并且以结果文件为准，不以退出码为准。

## B1：Isaac 实际生效的参数

`scenarios/capture_isaac_runtime.py`，按 Phase 1 `step6_3.py` 的场景读取实际生效的值。18 个字段读到 16 个。

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 求解器 | TGS，PCM 碰撞，patch 摩擦 | 显式设置 |
| 探头、方块的材料 | 静摩擦 0.2，动摩擦 1.0，恢复系数 0 | Isaac 构造器默认值 |
| 地面材料 | 静 0.5，动 0.5，恢复系数 0.8 | 显式设置 |
| 迭代次数 | 位置 16，速度 1 | USD 回退值 |
| contact / rest offset | −inf，即由 PhysX 自动决定 | USD 回退值 |
| 摩擦、恢复系数的组合模式 | 未读到，按 PhysX 默认的 average | — |

脚本本身修了三处：`sleep_enabled` 读错了属性；`close()` 结束进程导致文件没写出来；USD 四元数无法写进 JSON。

## B2：Isaac 闭环适配器

`scenarios/isaac_replay.py`。**Phase 2 的控制器原样接入**，一步反馈延迟和 0.05 N 采样阈值都保持一致，只把物理步换成 PhysX。所以记录里的差异只来自物理后端，不来自控制器。

驱动方式的对应：内核每个控制步覆写一次速度，并在每个子步抵消重力；Isaac 一侧用同样的速度覆写，同时关闭探头重力、锁定转动。

两组参数：

- **aligned**：零摩擦、零恢复系数、rest offset 0、contact offset 1 mm。比的是求解器本身。
- **phase1_defaults**：Phase 1 当时实际用的 Isaac 默认参数。比的是参数会带来多大影响。

## B3：结果

| 门槛 | K1 aligned | K2 aligned | K1 defaults | K2 defaults |
| --- | --- | --- | --- | --- |
| 接触起始 | ✅ 0 步 | ✅ 0 步 | ✅ −1 步 | ✅ −1 步 |
| 接触点 | ⚠️ 0.83 mm | ⚠️ 0.83 mm | ❌ 0.32 mm | ❌ 49 mm |
| 接触法线 | ✅ 0° | ✅ 0° | ✅ | ✅ |
| 动量平衡 | ✅ | ✅ | ✅ | ✅ |
| 力控稳态 | — | ✅ 两边都在第 322 帧进入 | — | ✅ |
| 逐帧法向力最大差（只报告） | 1.3e-8 N | 1.6e-7 N | 0.58 N | 0.98 N |
| 帧数、相位序列 | 相同 | 相同 | 差 1 帧 | 相同 |

### 接触点 0.83 mm 是约定差异，不是误差

| 阶段 | 接触点差 |
| --- | --- |
| 静止接触（446 帧） | 0 |
| 以 0.05 m/s 滑动 | 恒为 0.8333 mm，正好是 0.05 m/s × 1/60 s |
| 滑动时 Isaac 第 n 步对内核第 n−1 步 | 4.3×10⁻⁵ mm |

PhysX 在一步开始、积分之前生成接触，内核在一步结束时报告接触。

**用户裁决（2026-11-25）：选 (a)，记为已声明的约定差异，Phase 2 判定通过。** 0.1 mm 的阈值没有改。回归测试里按这个约定比较：滑动时用 Isaac 第 n 步对内核第 n−1 步。

### defaults 组的 49 mm

Isaac 默认的动摩擦 1.0 在滑动中拖住了探头，这是参数造成的差异。它说明 **Phase 1 用的默认参数并不中性**：如果当初直接把 Isaac 当真值，就会把参数带来的偏差当成物理结论。

## 落地

| 改动 | 位置 |
| --- | --- |
| `isaac_sim_parity` 标为已交付，K2-A 升为 **K2 完成** | `validation/k1_acceptance.py` |
| 参照存档（4 份闭环记录 + B1 参数，gzip 压缩，SHA-256 锁定） | `experiments/isaac_reference/` |
| 自动回归：内核一改，就和 Isaac 存档比；存档本身被改也会报错 | `validation/tests/test_isaac_reference.py`（9 个测试） |
| 比对工具 | `scenarios/isaac_replay.py`、`validation/isaac_parity.py`、`validation/isaac_smoke.py` |

## 不能过度解释的部分

1. **范围**：只覆盖单个球形探头、刚体、零摩擦、法向接触的 K1/K2。盒形探头、多探头、子步细分都没有比。
2. **摩擦与切向力没有比。** PhysX 的接触报告只给法向冲量，Isaac 一侧记录的切向力 0 不是测量值。
3. **PhysX 不是触觉的真值。** 这次证明的是"我们的求解器在刚体假设下算得对"，不证明刚体假设符合真实触觉。层 2、3 的真值来自 Phase 3 的真实指尖数据。
