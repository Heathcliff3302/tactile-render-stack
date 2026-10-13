"""Curve report: the run's own arrays, drawn, in one self-contained file.

A table of gate verdicts says whether a run passed. It does not show what the
run *did* — when contact started, how the regulator approached its target,
whether it sat in the band or drifted across it. Those are visible in a
second from a curve and effectively invisible in a column of numbers, and
they are what a reader needs to judge whether a passing run passed for the
right reason.

The plots are inline SVG generated from the arrays. No plotting dependency is
added: the repository depends on ``jsonschema`` and ``numpy`` and a report is
not a reason to grow that list, especially one that must stay runnable on a
machine with no display. The output is a single HTML file that opens offline
with no server and no assets beside it.

Every series is read from the run itself, never recomputed from a model. The
regulation band is drawn from the declared ``target_force_n`` and
``force_tolerance_n``, so what the reader sees is the band the gate scored
against, not a redrawn approximation of it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from contact_kernel import RigidContactKernel
from virtual_reality import PHASE_ROLES

#: Fixed plot geometry, in user units of the SVG viewBox.
WIDTH = 920
HEIGHT = 260
PAD_LEFT = 72
PAD_RIGHT = 24
PAD_TOP = 20
PAD_BOTTOM = 42

#: One colour per motion role, so the segment bands read the same way across
#: trajectory and force runs even though the segments are named differently.
ROLE_COLOURS = {
    "pre_contact": "#eef1f5",
    "approach": "#e3eefc",
    "hold": "#e6f5ea",
    "slide": "#fdf2dc",
    "retract": "#f6e9f2",
    "released": "#f0f0f0",
}


def _scale(value, lo, hi, out_lo, out_hi):
    if hi <= lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def _ticks(lo, hi, count=5):
    if hi <= lo:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + step * index for index in range(count)]


def _format(value):
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    if magnitude >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def _segment_bands(times, phases, x_lo, x_hi, plot_left, plot_right):
    """Shaded background spans, one per contiguous segment."""
    if not len(times):
        return "", []
    bands, labels = [], []
    start = 0
    for index in range(1, len(phases) + 1):
        if index < len(phases) and phases[index] == phases[start]:
            continue
        phase = phases[start]
        x0 = _scale(times[start], x_lo, x_hi, plot_left, plot_right)
        x1 = _scale(
            times[min(index, len(times) - 1)], x_lo, x_hi, plot_left, plot_right
        )
        colour = ROLE_COLOURS.get(PHASE_ROLES.get(phase, ""), "#f4f4f4")
        bands.append(
            f'<rect x="{x0:.1f}" y="{PAD_TOP}" width="{max(x1 - x0, 0.5):.1f}" '
            f'height="{HEIGHT - PAD_TOP - PAD_BOTTOM}" fill="{colour}"/>'
        )
        labels.append((phase, x0, x1))
        start = index
    return "".join(bands), labels


def _polyline(times, values, x_lo, x_hi, y_lo, y_hi, left, right, colour, width=1.8):
    """One polyline per run of defined values.

    A frame with no contact has no force, and joining across it would draw a
    straight segment through a gap the run never produced — the picture would
    assert continuity the data denies. So an undefined value ends the current
    line and the next defined value starts a new one.
    """
    runs, current = [], []
    for time_s, value in zip(times, values):
        if value is None:
            if current:
                runs.append(current)
                current = []
            continue
        x = _scale(time_s, x_lo, x_hi, left, right)
        y = _scale(value, y_lo, y_hi, HEIGHT - PAD_BOTTOM, PAD_TOP)
        current.append(f"{x:.1f},{y:.1f}")
    if current:
        runs.append(current)
    return "".join(
        f'<polyline fill="none" stroke="{colour}" stroke-width="{width}" '
        f'stroke-linejoin="round" points="{" ".join(points)}"/>'
        for points in runs
    )


def _chart(title, y_label, times, series, phases=(), *, bands=None,
           band_labels=True, extra_shapes="", y_floor=None):
    """One labelled SVG chart. ``series`` is a list of (label, values, colour)."""
    plot_left, plot_right = PAD_LEFT, WIDTH - PAD_RIGHT
    x_lo = float(times[0]) if len(times) else 0.0
    x_hi = float(times[-1]) if len(times) else 1.0

    finite = [value for _, values, _ in series for value in values if value is not None]
    if bands:
        finite.extend(bands)
    y_lo = min(finite) if finite else 0.0
    y_hi = max(finite) if finite else 1.0
    if y_floor is not None:
        y_lo = min(y_lo, y_floor)
    span = y_hi - y_lo
    pad = span * 0.08 if span else max(abs(y_hi), 1.0) * 0.1
    y_lo, y_hi = y_lo - pad, y_hi + pad

    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(title)}">']
    band_svg, labels = _segment_bands(times, phases, x_lo, x_hi, plot_left, plot_right)
    parts.append(band_svg)

    for value in _ticks(y_lo, y_hi):
        y = _scale(value, y_lo, y_hi, HEIGHT - PAD_BOTTOM, PAD_TOP)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{_format(value)}</text>'
        )
    for value in _ticks(x_lo, x_hi):
        x = _scale(value, x_lo, x_hi, plot_left, plot_right)
        parts.append(
            f'<text x="{x:.1f}" y="{HEIGHT - PAD_BOTTOM + 18}" text-anchor="middle" '
            f'class="tick">{_format(value)}</text>'
        )
    parts.append(extra_shapes)
    for label, values, colour in series:
        parts.append(_polyline(times, values, x_lo, x_hi, y_lo, y_hi,
                               plot_left, plot_right, colour))
    if band_labels:
        for phase, x0, x1 in labels:
            if x1 - x0 < 46:
                continue
            parts.append(
                f'<text x="{(x0 + x1) / 2:.1f}" y="{PAD_TOP + 13}" '
                f'text-anchor="middle" class="band">{html.escape(phase)}</text>'
            )
    parts.append(
        f'<text x="{plot_left}" y="{HEIGHT - 6}" class="axis">时间 time_s</text>'
        f'<text x="14" y="{PAD_TOP + 4}" class="axis">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{colour}"></i>{html.escape(label)}</span>'
        for label, _, colour in series
    )
    return (
        f'<section class="chart"><h3>{html.escape(title)}</h3>'
        f'<div class="legend">{legend}</div>{"".join(parts)}</section>'
    )


def _collect(run):
    """Per-frame series, taken from the run rather than recomputed."""
    config = run.kernel_config
    key = RigidContactKernel.pair_key(config.probes[0].id, config.surface.id)
    times, phases, heights, commands = [], [], [], []
    normal_force, tangent_speed, active_cells, area = [], [], [], []
    measured, target = [], []
    for loop_frame in run.loop_frames:
        frame = loop_frame.interaction
        pair = loop_frame.world.kernel_frame.pairs[key]
        times.append(loop_frame.time_s)
        phases.append(loop_frame.phase)
        heights.append(frame.metadata["probe_position_world_m"][2])
        commands.append(loop_frame.world.command_velocity_world_mps[config.probes[0].id][2])
        normal_force.append(frame.normal_force_n)
        tangent = frame.tangent_velocity_world_mps
        tangent_speed.append(abs(tangent[0]) if tangent is not None else None)
        active_cells.append(len(pair.cells))
        area.append(frame.contact_area_proxy_m2)
        measured.append(loop_frame.measurement.measured_force_n)
        target.append(loop_frame.target.normal_force_n)
    return {
        "time_s": times, "phase": phases, "probe_height_m": heights,
        "command_vz_mps": commands, "normal_force_n": normal_force,
        "tangent_speed_mps": tangent_speed, "active_cells": active_cells,
        "contact_area_proxy_m2": area, "measured_force_n": measured,
        "target_force_n": target,
    }


def _band_shapes(series, force_config, x_lo, x_hi, y_lo, y_hi):
    """The declared tolerance band, drawn as the gate scored it."""
    if not force_config:
        return "", []
    target = force_config["target_force_n"]
    tolerance = force_config["force_tolerance_n"]
    plot_left, plot_right = PAD_LEFT, WIDTH - PAD_RIGHT
    top = _scale(target + tolerance, y_lo, y_hi, HEIGHT - PAD_BOTTOM, PAD_TOP)
    bottom = _scale(target - tolerance, y_lo, y_hi, HEIGHT - PAD_BOTTOM, PAD_TOP)
    middle = _scale(target, y_lo, y_hi, HEIGHT - PAD_BOTTOM, PAD_TOP)
    return (
        f'<rect x="{plot_left}" y="{top:.1f}" width="{plot_right - plot_left}" '
        f'height="{max(bottom - top, 1):.1f}" fill="#2f9e4420" stroke="none"/>'
        f'<line x1="{plot_left}" y1="{middle:.1f}" x2="{plot_right}" y2="{middle:.1f}" '
        f'stroke="#2f9e44" stroke-width="1.2" stroke-dasharray="5 4"/>'
    ), [target - tolerance, target + tolerance]


def build_report(run, report: dict, effective_runtime: dict) -> str:
    """The complete HTML document for one run."""
    series = _collect(run)
    data = run.spec.to_dict()
    force_config = data["controller"]["force"]
    times = series["time_s"]
    phases = series["phase"]

    def chart(title, y_label, items, **kwargs):
        return _chart(title, y_label, times, items, phases, **kwargs)

    charts = [chart(
        "探头高度与法向命令速度 · probe height and commanded normal speed",
        "m  |  m/s",
        [("probe_height_m", series["probe_height_m"], "#1c7ed6"),
         ("command_vz_mps", series["command_vz_mps"], "#e8590c")],
    )]

    force_values = [value for value in series["normal_force_n"] if value is not None]
    band_shapes, band_values = "", None
    if force_config and force_values:
        lo = min(force_values + [0.0])
        hi = max(force_values + [force_config["target_force_n"] + force_config["force_tolerance_n"]])
        span = hi - lo
        pad = span * 0.08 if span else 0.1
        band_shapes, band_values = _band_shapes(
            series, force_config, times[0], times[-1], lo - pad, hi + pad
        )
    charts.append(chart(
        "法向力与调节带 · normal force against the declared band" if force_config
        else "法向力 · normal force",
        "N",
        [("normal_force_n (sampled)", series["normal_force_n"], "#1c7ed6"),
         ("measured_force_n (layer 5)", series["measured_force_n"], "#862e9c")],
        bands=band_values, extra_shapes=band_shapes, y_floor=0.0,
    ))
    charts.append(chart(
        "切向速度 · tangential speed",
        "m/s",
        [("tangent_speed_mps", series["tangent_speed_mps"], "#0c8599")],
        y_floor=0.0,
    ))
    charts.append(chart(
        "接触单元数与占格代理面积 · active cells and grid proxy area",
        "cells  |  m²",
        [("active_cells", [float(value) for value in series["active_cells"]], "#5c940d"),
         ("contact_area_proxy_m2", series["contact_area_proxy_m2"], "#d6336c")],
        y_floor=0.0,
    ))

    gates = "".join(
        f'<tr class="{"pass" if gate["passed"] else "fail"}">'
        f'<td>{"通过" if gate["passed"] else "未通过"}</td>'
        f'<td><code>{html.escape(gate["gate"])}</code></td></tr>'
        for gate in report["gates"]
    )
    milestone = report["milestone"]
    outstanding = "".join(
        f"<li><code>{html.escape(name)}</code> — "
        f'{html.escape(milestone["deliverables"][name]["note"])}</li>'
        for name in milestone["outstanding"]
    )
    segments = "".join(
        f"<tr><td><code>{html.escape(phase)}</code></td><td>{count}</td>"
        f"<td>{count * run.kernel_config.timing.output_dt_s:.3f}</td></tr>"
        for phase, count in report["segments"].items()
    )
    authority = effective_runtime.get("realised_drive_authority")
    authority_rows = ""
    if authority:
        authority_rows = (
            f"<tr><td>实测力上限 measured ceiling</td>"
            f'<td>{authority["measured_force_ceiling_n"]:.4f} N</td></tr>'
            f"<tr><td>理想律预测 ideal law</td>"
            f'<td>{authority["ideal_law_force_ceiling_n"]:.4f} N</td></tr>'
        )

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(run.run_id)} · 接触曲线报告</title>
<style>
  :root {{
    --bg:#ffffff; --fg:#1a1d21; --muted:#5f6771; --line:#d9dee5; --grid:#edf0f4;
    --pass:#2f9e44; --fail:#c92a2a; --card:#fbfcfd;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#14171a; --fg:#e8eaed; --muted:#9aa3ad; --line:#2c3238; --grid:#20252a;
      --card:#191d21;
    }}
  }}
  body {{ background:var(--bg); color:var(--fg); margin:0;
    font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }}
  main {{ max-width:1000px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:34px 0 12px; padding-bottom:6px;
       border-bottom:1px solid var(--line); }}
  h3 {{ font-size:14px; font-weight:600; margin:0 0 2px; }}
  .sub {{ color:var(--muted); margin:0 0 20px; }}
  .chart {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
    padding:14px 16px 6px; margin-bottom:16px; overflow-x:auto; }}
  .chart svg {{ width:100%; height:auto; display:block; }}
  .legend {{ margin:0 0 6px; color:var(--muted); font-size:12px; }}
  .key {{ margin-right:14px; white-space:nowrap; }}
  .key i {{ display:inline-block; width:11px; height:3px; border-radius:2px;
    vertical-align:middle; margin-right:5px; }}
  text.tick {{ font-size:10px; fill:var(--muted); }}
  text.axis {{ font-size:11px; fill:var(--muted); }}
  text.band {{ font-size:10px; fill:var(--muted); }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:8px; }}
  td, th {{ border-bottom:1px solid var(--line); padding:6px 10px; text-align:left; }}
  tr.pass td:first-child {{ color:var(--pass); }}
  tr.fail td:first-child {{ color:var(--fail); font-weight:600; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:20px; }}
  .note {{ color:var(--muted); font-size:13px; border-left:3px solid var(--line);
    padding-left:12px; margin:14px 0; }}
</style></head><body><main>

<h1>{html.escape(run.run_id)} · 接触曲线报告</h1>
<p class="sub">
  {html.escape(run.scenario_id)} · {html.escape(run.controller_mode)} 控制 ·
  {len(run.frames)} 帧 / {run.end_time_s:.3f} s ·
  后端 <code>{html.escape(run.kernel_config.backend_id)}</code>
</p>

<h2>曲线</h2>
{"".join(charts)}
<p class="note">
  每条曲线直接取自本次运行的数组，不由模型重算。调节带画的是声明的
  <code>target_force_n ± force_tolerance_n</code>，即门槛评分所依据的那条带。
  背景分段底色按运动角色着色，所以轨迹模式与力控模式的同一角色颜色一致。
</p>

<h2>验收与里程碑</h2>
<div class="cols">
  <div>
    <h3>门槛 {sum(g["passed"] for g in report["gates"])} / {len(report["gates"])}</h3>
    <table>{gates}</table>
  </div>
  <div>
    <h3>里程碑 {html.escape(milestone["milestone"])}</h3>
    <table>
      <tr><td>本次运行验收</td><td><code>{str(report["run_acceptance_passed"]).lower()}</code></td></tr>
      <tr><td>里程碑完成</td><td><code>{str(milestone["milestone_complete"]).lower()}</code></td></tr>
      <tr><td>子里程碑</td><td><code>{html.escape(milestone["sub_milestone"])}</code></td></tr>
      {authority_rows}
    </table>
    {"<p class='note'>尚缺：</p><ul>" + outstanding + "</ul>" if outstanding else ""}
  </div>
</div>

<h2>分段</h2>
<table><tr><th>段</th><th>帧</th><th>秒</th></tr>{segments}</table>

<h2>边界</h2>
<p class="note">
  <code>contact_area_proxy_m2</code> 是占格代理面积，随输出分辨率变化，不是物理接触面积；
  本次运行 <code>physical_contact_area_m2</code> 全程为 <code>unavailable</code>。
  层 5 的 <code>measured_force_n</code> 由声明的传递模型产生，
  <code>status = modelled_not_measured</code>，未接硬件。
  与 Isaac Sim 之间目前只有行为与守恒门槛，没有数值等价结论。
</p>

<p class="sub">
  conditions_sha256 <code>{html.escape(run.conditions_sha256)}</code><br>
  spec_sha256 <code>{html.escape(data["spec_sha256"])}</code>
</p>

</main></body></html>
"""


def write_curve_report(path, run, report: dict, effective_runtime: dict) -> dict:
    """Write the report and return a small record of what it contains."""
    path = Path(path)
    path.write_text(build_report(run, report, effective_runtime),
                    encoding="utf-8", newline="\n")
    return {
        "schema": "curve-report/v1",
        "path": path.name,
        "frames": len(run.loop_frames),
        "charts": [
            "probe height and commanded normal speed",
            "normal force against the declared band",
            "tangential speed",
            "active cells and grid proxy area",
        ],
        "source": "the run's own per-frame arrays, not recomputed from a model",
    }
