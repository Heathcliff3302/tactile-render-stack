# tactile-render-stack

`tactile-render-stack` is the second phase of the Virtual Immersion roadmap. It decouples the tactile rendering pipeline from Isaac Sim: a self-contained CPU contact kernel replaces the simulator as the authoritative physics source, and a five-layer loop carries every contact from virtual-world state to an actuator command under versioned, machine-checked contracts.

Phase 1, [isaacsim-rigid-contact-pipeline](https://github.com/Heathcliff3302/isaacsim-rigid-contact-pipeline), established the rigid-contact baseline in Isaac Sim and is retained as reference evidence. This repository reproduces that baseline without a simulator and validates the result against Isaac Sim numerically.

## Status

Phase 2 was closed on 2026-11-25.

| Milestone | Scope | Status |
| --- | --- | --- |
| K0 Baseline and experiment contracts | Phase 1 parameter provenance, CPU reference specifications, JSON Schemas, comparison profile | Complete |
| K1 Single-probe rigid mode | Decoupled kernel, five-layer loop, 10 acceptance gates, runtime field verdicts | Complete, frozen |
| K2 Force control and regression | PI normal-force regulation to 1 N, 13-channel dense field, curve report, 13 acceptance gates, Isaac Sim parity | Complete (2026-11-25) |
| P2.2 Multi-probe and resolution | P2.2.1 box probe with multi-cell contact; P2.2.2 independent per-pair streams; P2.2.3 resolution regression | P2.2.1 complete; P2.2.2 implemented, pending acceptance; P2.2.3 frozen |
| P2.4 IMU texture spectrum | Spectral analysis for an external vibration reference | Analysis implemented; no data captured; retained as a side track |
| K4 Material response | Hertz, elastic foundation, viscoelastic models | Transferred to Phase 3 as the physics baseline |
| K5 Actuator loop | Hardware actuator calibration and closed loop | Out of scope; layer 5 is delegated to hardware vendors |

Isaac Sim parity, measured on the aligned-parameter K1 and K2 runs against Isaac Sim 5.1.0:

| Quantity | Result |
| --- | --- |
| Per-frame normal force, maximum difference | 1.3 × 10⁻⁸ N (K1), 1.6 × 10⁻⁷ N (K2) |
| Contact onset step, contact normal, phase sequence, frame count | Identical |
| Force steady state (±5 %, last 30 frames) | Both enter at frame 322 and hold |
| Contact point | Identical in static contact; one control step apart while sliding, a declared reporting convention (PhysX reports contacts at step start, the kernel at step end) |

The full record, including scope and the quantities that were not compared, is in [`docs/step_record/17-Phase2收尾-Isaac比对.md`](docs/step_record/17-Phase2收尾-Isaac比对.md). The phase overview is [`docs/step_record/10-Phase2总推进.md`](docs/step_record/10-Phase2总推进.md).

## Table of Contents

- [Status](#status)
- [Scope](#scope)
- [Repository Structure](#repository-structure)
- [Five-Layer Pipeline](#five-layer-pipeline)
- [Contact Data Sources](#contact-data-sources)
- [Usage](#usage)
- [Validation](#validation)
- [Evidence Rules](#evidence-rules)
- [Boundaries](#boundaries)
- [Documentation](#documentation)

## Scope

The phase answers three questions:

1. Can the Phase 1 rigid-contact baseline be reproduced without a simulator, on a kernel that runs on CPU and depends only on the standard library?
2. Can layer 2 become an open boundary, so that any contact source producing valid `InteractionState v2` records drives layers 3 to 5 unchanged?
3. Does the kernel agree numerically with Isaac Sim under the same physical assumptions?

The kernel solves one rotation-locked sphere or box probe against one fixed plane, with zero friction and zero restitution. It is a reference implementation for these assumptions, not a general rigid-body solver.

## Repository Structure

```text
├── virtual_reality/     // Layer 1: world state, trajectory and force state machines, fixed clock, field verdicts
├── interaction/         // Layer 2: adapters from contact sources to InteractionState
├── mechanics/           // Layer 3: material response models and experiment assembly
├── std_tactile/         // Layer 4: tactile targets, target sequences, replay
├── actuators/           // Layer 5: actuator commands, inverse model, measurement, watchdog
├── contact_kernel/      // CPU contact kernel; standard library only; not part of any layer
├── tactile_contract/    // Cross-layer contracts, JSON Schemas, serialisation and replay
├── scenarios/           // Experiment specifications, parameter inventories, Isaac Sim tools
├── validation/          // Acceptance gates, cross-backend comparison, Isaac Sim parity
├── experiments/         // Run indexes and the Isaac Sim reference records
├── docs/                // Design notes, plans and step records
└── pyproject.toml       // Package metadata
```

The directory layout is stable across phases. Later phases add backends, models, scenarios and acceptance records; they do not introduce a parallel top-level architecture.

## Five-Layer Pipeline

```text
1. virtual-world state     pose, velocity, trajectory command, fixed simulation clock
2. contact detection       contact points, normal, gap, relative velocity, contact region  -> InteractionState
3. material response       stiffness, damping, viscoelasticity, contact area, pressure     -> MaterialResponse
4. tactile target          device-independent force, vibration and temperature targets     -> TactileTarget
5. actuator inverse model  range, bandwidth, non-linearity, latency, watchdog              -> ActuatorCommand
```

Each control step runs all five layers in order and must produce a record from each; a missing record raises. Layer 1 owns the kernel instance and is the only stage that advances physics. Layer 2 reads and formats the kernel frame and never steps the solver, so each contact pair has exactly one authoritative dynamics solver per step. Controllers read feedback with an explicit one-step delay.

Human perception is the acceptance target at the output, not a layer to be built.

| Layer | Implementation | Limitation |
| --- | --- | --- |
| 1 | Trajectory and force state machines, runtime field verdicts, command limits, hard `max_duration_s` termination, load-time force-control checks | Single sphere, rotation locked, one fixed plane |
| 2 | Point → Cell → Patch → Episode, 13-channel dense field, four area layers, versioned `InteractionState v2` | Area is an occupancy proxy; `physical_contact_area_m2` is `unavailable` |
| 3 | `RigidBaselineMaterial`, `status = passthrough` | Stiffness and damping are `null` |
| 4 | `NormalForceTargetStage`, one channel | Vibration and temperature are `null` |
| 5 | Single-axis command path, inverse model, saturation, latency, watchdog | `status = modelled_not_measured`; no hardware |

## Contact Data Sources

| Source | Status |
| --- | --- |
| CPU contact kernel, `contact_kernel/` | Implemented; standard library only |
| Isaac Sim 5.1.0 | Compared (2026-11-25); four closed-loop records stored as a regression reference |
| Unity scene input | Not implemented; Unity physics is also PhysX, so an integration must keep the backend identifier to avoid a second solver |
| Real fingertip data | Phase 3 |

## Usage

### Prerequisites

- Python 3.10 or later.
- No simulator and no GPU. `contact_kernel/` depends only on the standard library; the rest depends on `numpy` and `jsonschema`.
- Isaac Sim is needed only to recapture the reference records; see [Isaac Sim tools](#isaac-sim-tools).

### Installation

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"     # Linux/macOS: .venv/bin/python
```

### Running

```bash
.venv/Scripts/python.exe -m mechanics.run_k1            # K1 trajectory run
.venv/Scripts/python.exe -m mechanics.run_k2            # K2 force-control run
.venv/Scripts/python.exe -m mechanics.run_k3            # P2.2.1 box probe on the kernel
.venv/Scripts/python.exe -m mechanics.run_k3_multi      # P2.2.2 multi-probe streams
.venv/Scripts/python.exe -m pytest -q                   # 325 tests, including the Isaac Sim regression
```

Run commands from the repository root.

### Output

Each run writes a full run directory to `experiments/runs/<run_id>/`, which is not version-controlled: the experiment manifest, per-layer record streams, sparse points, cells and patches, the dense field `dense_contact_field.npz`, the effective runtime record, the acceptance report and `contact_report.html`. A compact index is written to `experiments/index/` and is version-controlled; it holds the specification, condition and manifest hashes, the gate results and the SHA-256 of every artifact.

### Isaac Sim Tools

These run on a GPU machine with Isaac Sim 5.1.0 and are not needed for normal use.

| Tool | Purpose |
| --- | --- |
| `validation/isaac_smoke.py` | Headless start, PhysX step and RTX render check |
| `scenarios/capture_isaac_runtime.py` | Reads the effective PhysX settings of the Phase 1 scene |
| `scenarios/isaac_replay.py` | Drives PhysX with this repository's controller and writes a closed-loop record |
| `validation/isaac_parity.py` | Scores the kernel against Isaac Sim records |
| `docs/autodl_isaac_setup.sh` | Environment setup for an AutoDL container |

The NVIDIA driver must be 580.x. On 595.71.05 Isaac Sim 5.1.0 terminates with a segmentation fault in `librtx.scenedb.plugin` at renderer start-up.

## Validation

### K1 Acceptance Gates

| Gate | Result |
| --- | --- |
| `run_completes_declared_program` | `completed`, no abort, all segments present |
| `single_probe_contact_appears` | 121 contact frames, peak 0.6 N |
| `hold_contact_is_stable` | Coverage 1.0, tangential speed 0.0 m/s |
| `contact_coverage_meets_threshold` | 1.0 (threshold 0.95) |
| `slide_speed_meets_threshold` | 1.0 of samples (threshold 0.90, tolerance ±0.01 m/s) |
| `release_clears_contact_force` | No contact frames, peak force 0.0 N |
| `force_matches_impulse_and_is_conserved` | Impulse, normal-plus-tangent and Point→Cell→Patch errors 0.0 N (threshold 10⁻⁵ N) |
| `time_step_does_not_change_data_semantics` | Segment, contact, cell and patch sequences identical at 1, 2 and 4 substeps |
| `dense_field_matches_sparse_cells` | No unmapped or extra cells; per-cell force error 3 × 10⁻⁸ N |
| `interaction_state_replays_completely` | 18 artifact hashes and 472 frames validated against the contract |

The K1 run has 472 frames (7.87 s): `pre_contact 30 | approach 175 | hold 60 | lateral_slide 60 | retract 117 | released 30`.

### K2 Acceptance Gates

K2 adds three force gates to the K1 set: `force_target_is_reached_and_held`, `force_held_through_slide` and `regulator_stays_within_declared_authority`. The K2 run has 797 frames.

### Isaac Sim Regression

`validation/tests/test_isaac_reference.py` reruns K1 and K2 on every test run and compares them with the stored Isaac Sim records in `experiments/isaac_reference/`. It requires per-frame normal force within 10⁻⁶ N, every frozen gate passing, and contact points within 0.1 mm under the declared convention. The records are locked by SHA-256; replacing them requires a new capture.

## Evidence Rules

The rules below are enforced in code. Their purpose is to make an apparent pass and a real pass impossible to confuse.

| Rule | Requirement | Enforced in |
| --- | --- | --- |
| Unavailable values | A quantity the source cannot produce is recorded as `unavailable` or `null`, never as zero | `interaction/kernel_backend.py` |
| One method per observable | Each observable has one computation method for the whole run, bound to the experiment manifest | `interaction/kernel_backend.py`, `tactile_contract/experiment.py` |
| Area layers | Occupancy proxy, swept path, geometric footprint and model footprint are reported separately, each with its method name | `contact_kernel/kernel.py` (`AreaLayers`) |
| Threshold ledger | Every declared threshold is consumed by a gate or registered as deferred; any other key is rejected | `validation/k1_acceptance.py` |
| Field verdicts | Every runtime field is `enforced`, `recorded_inert` or `single_implementation`; unsupported values are rejected at load time | `virtual_reality/runtime_policy.py` |
| Force-control limits | Drive clamp, integral authority and proportional stability are checked independently at load time | `virtual_reality/drive_limits.py` |
| Measured drive authority | The closed-form limits are confirmed by applying the declared press limit on the kernel | `virtual_reality/authority_probe.py` |
| Duration and fraction | Hold and slide gates require the declared duration, not only an in-band fraction | `validation/k1_acceptance.py` |
| Run versus milestone | `run_acceptance_passed` and `milestone_complete` are separate fields; the latter comes from a deliverable ledger | `validation/k1_acceptance.py` (`milestone_status`) |
| Hash binding | The manifest binds conditions, observable definitions and the SHA-256 of every artifact; replay verifies each | `tactile_contract/experiment.py` |
| Realised commands | Feedback-driven commands are recorded in a first pass and stamped in a second; both passes must match exactly | `mechanics/k1_rigid.py` |
| Frozen comparison thresholds | Cross-backend thresholds are code constants set before the comparison and are not changed to make a run pass | `validation/isaac_parity.py` (`THRESHOLDS`) |

### Force-Control Limits

The three force-control limits are independent; satisfying one does not imply the others.

| Limit | Condition | Constrains |
| --- | --- | --- |
| Drive clamp | `max_press_speed × mass / output_dt ≥ F_target + tolerance` | Whether a large enough command can be issued |
| Integral authority | `force_ki × force_integral_limit ≥ F_target × output_dt / mass` | Whether the command can be sustained in steady state |
| Proportional stability | `mass × force_kp / output_dt < 1` | Whether the regulator converges |

Two cases recorded during K2 show the independence. The Phase 1 value of `force_ki` passes the drive clamp but stalls at 0.75 N and times out at 20 s. `force_kp = 0.2` is stable at the declared 60 Hz and oscillates at 120 Hz although its authority is sufficient.

The closed-form conditions assume an ideal drive. With `linear_damping_per_s = 30`, all three pass and predict 1.5 N while the kernel delivers 0.75 N. The measured authority probe therefore applies the declared press limit on the kernel and reads the reported force. It covers the open-loop ceiling only, not feedback delay or PI gains.

## Boundaries

- The kernel solves a rotation-locked sphere or box against one fixed plane, with zero friction and zero restitution. Unsupported configurations raise an error that names the milestone responsible, rather than being approximated.
- Isaac Sim parity covers the aligned-parameter K1 and K2 runs only: rigid, single sphere, zero friction, normal contact. Friction and tangential force were not compared, because the PhysX contact report carries normal impulse only. Box and multi-probe runs were not compared.
- PhysX is not a ground truth for touch. Parity shows that the solver is correct under rigid assumptions, not that rigid assumptions describe real contact.
- Layers 3, 4 and 5 use in-process interfaces with units, model identifiers and status flags. They are not versioned contracts yet.
- Layer 5 has no hardware. Measurements come from a declared transfer model and are marked `status = modelled_not_measured`.
- Regenerating the K0 inputs (`python -m scenarios.build_k0`) requires the Phase 1 repository in a sibling directory. The generated inputs are committed, so normal use does not need it.

## Documentation

| Document | Content |
| --- | --- |
| [`docs/step_record/`](docs/step_record/) | Phase overview, per-milestone records, Isaac Sim parity record and follow-up list |
| [`docs/contact_kernel_design.md`](docs/contact_kernel_design.md) | Kernel design and external references: PhysX contact generation, Unity fixed-step simulation, Hertz contact, Drake hydroelastic contact |
| [`docs/k1_single_probe_rigid.md`](docs/k1_single_probe_rigid.md) | K1 specification and acceptance |
| [`docs/phase_boundary.md`](docs/phase_boundary.md) | Migration boundary from Phase 1 |
| [`docs/plans/`](docs/plans/) | K1 and K2 implementation plans |
| [`tactile_contract/`](tactile_contract/) | Contract specifications and JSON Schemas |

The long-term stack this repository belongs to:

```text
1. virtual-world state
2. contact detection and contact field
3. material response
4. tactile target representation
5. actuator inverse model and hardware output
```

Phase 2 provides layers 1 and 2 for rigid contact with a validated kernel, and placeholder records for layers 3 to 5. Phase 3 learns layers 2 and 3 from real fingertip data; layer 5 is delegated to hardware vendors.
