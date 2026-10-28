# 在租用的 GPU 机器上采集 Isaac Sim 生效参数

这是 K2 唯一需要 GPU 的一步，而且只需要几分钟。本页是给**在那台机器上操作的人**看的，不需要读懂 Phase 2 的其他部分。

## 为什么需要这一步

Phase 1 的 `scripts/step6_3.py` 用 `DynamicSphere` / `FixedCuboid` / `add_default_ground_plane()` 建场，**从未显式设置**摩擦、恢复系数、阻尼、求解器迭代次数、接触偏移。因此 K0 参数清单里有 18 个字段标为 `unresolved`——值确实存在，但只在引擎内部。

读到它们之前，Phase 2 与 Isaac Sim 之间只能有行为与守恒门槛，**没有数值等价结论**。这一条就是 K2-A 与 K2 之间的全部差距。

## 需要什么机器

| 项 | 要求 | Phase 1 当时用的 |
| --- | --- | --- |
| GPU | 支持 Isaac Sim 的 RTX 卡 | RTX 4090 |
| Isaac Sim | 5.x（能装上就行，版本会被记进结果） | 5.1.0 |
| Python | 用 Isaac Sim 自带的，不要用系统 Python | 3.11 |
| 显示器 | **不需要**，脚本默认无头运行 | — |
| 时长 | **约 5–10 分钟**（含启动） | — |

最省钱的做法是用官方容器镜像，省掉装环境的时间：

```bash
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v /path/to/capture_isaac_runtime.py:/capture.py \
  nvcr.io/nvidia/isaac-sim:5.1.0
```

镜像名与版本以 NVIDIA NGC 当前发布为准；Isaac Sim 4.x 也可以，版本差异会被记录下来，不影响采集本身。

## 要做的事：三步

**1. 把脚本拷到那台机器上。** 只需要这一个文件：

```
Phase_2/scenarios/capture_isaac_runtime.py
```

**2. 用 Isaac Sim 自带的 Python 跑它。**

```bash
# 容器里
/isaac-sim/python.sh /capture.py --output /isaac_runtime.json

# 本地安装，典型路径
~/.local/share/ov/pkg/isaac-sim-*/python.sh capture_isaac_runtime.py \
    --output isaac_runtime.json
```

必须用 `python.sh`（或 `kit/python`），不能用系统 `python3`——Isaac Sim 的模块只在它自己的解释器里。

**3. 把生成的 `isaac_runtime.json` 带回来。** 那台机器上不需要其他任何东西。

## 预期输出

脚本结束时会打印一段摘要，大致形如：

```json
{
  "expected_fields": 18,
  "fields_read": 14,
  "fields_not_located": [],
  "fields_still_unresolved": ["restitution_combine_mode", "..."],
  "authored": ["probe_mass_kg", "..."],
  "schema_fallback": ["linear_damping_per_s", "..."]
}
```

**`fields_still_unresolved` 不为空是正常的，不是失败。** 脚本绝不猜值：某个字段读不到就如实标为未解析，在 K0 清单里继续保持 `unresolved`。一份如实标注的半份采集是有用的；一份数字来源不明的"完整"采集比没有更糟，因为等价结论会建在它上面。

同样重要的是 `origin` 字段的区分：

- `authored` —— 属性在 prim 上有显式赋值，这是场景真正设定的值；
- `schema_fallback` —— 属性存在但未赋值，这是 USD schema 的回退值。**证据强度较弱**：PhysX 可能使用它自己的内部默认值，而 USD 并不暴露那个值；
- `absent` / `error` —— 读不到，记为 `null`，绝不记成 0。

脚本还会导出一份 `attribute_inventory`，把探头、立方体、物理场景和材料 prim 上**所有**物理相关属性全部罗列。这样即使我某个字段找错了位置，也能从这份采集里补回来，**不必再租一次机器**。这是刻意为之的冗余。

## 如果失败了

脚本把异常写进 JSON 的 `capture_errors`，并且仍然把已读到的部分写出来。**即使报错也请把 JSON 带回来**——里面的 `attribute_inventory` 往往已经够用了。

常见情况：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: isaacsim` | 用了系统 Python | 换 `python.sh` |
| 启动卡住几分钟 | Isaac Sim 首次启动要编译 shader cache | 正常，等它 |
| `[Error] ... GPU` | 卡不被支持或驱动不匹配 | 把完整报错发我 |
| 某些字段 `absent` | 场景确实没设那个属性 | 正常，见上 |

## 拿到 JSON 之后（我这边做，不需要 GPU）

1. 把采集值写回 `scenarios/k0/step6_*.json`，`status` 从 `unresolved` 改为 `source_captured`，并带上 `origin` 与读取路径作为 evidence。
2. 重新评估比较门槛：`spec_comparison_gate` 目前因 `unresolved` 字段直接判不可比，解析后才可能进入数值比较。
3. 写 Isaac Sim → `InteractionState v2` 适配器。
4. 两个后端跑同一场景，做数值比较。

第 1、2、3 步都不需要 GPU。只有第 4 步需要，而且到那时我已经知道两边参数差在哪，可以先把比较脚本写好，再一次性用掉 GPU 时间。

## 边界

这一步只解决"PhysX 实际用了什么参数"。它**不**构成两个后端的等价结论——那需要第 4 步的实际比较。在此之前，仓库里的表述仍然是：**解耦已完成，等价未验证。**
