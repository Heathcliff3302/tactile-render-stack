# Phase 1 And Phase 2 Boundaries

## Phase 1

Phase 1 ends with the Isaac Sim rigid baseline. Its scope includes rigid world state, trajectory control, force feedback, contact extraction, contact patches, dense replay, and regression checks.

The Phase 1 repository remains an evidence source. Its output is migrated at the boundary into `InteractionState v2` with an `ExperimentManifest`.

## Phase 2

Phase 2 adds a platform-independent material response layer and a minimum five-layer tactile loop. The five layer directories remain stable for later phases.

| Layer | Phase 2 completion evidence |
| --- | --- |
| Layer 1 | A declared trajectory and world input can be replayed from a manifest |
| Layer 2 | Isaac Sim and a reduced kernel emit validated `InteractionState` frames |
| Layer 3 | Hard, Kelvin–Voigt, and Standard Linear Solid responses have repeatable fixtures |
| Layer 4 | A target sequence can be generated, stored, replayed, and scored |
| Layer 5 | One actuator has a command path, measurement path, inverse model, and watchdog behavior |

## Phase comparison rule

Phase 3 keeps this directory layout. New models and backends add versioned subdirectories. A phase comparison links the same scenario ID to its manifest, contract version, backend version, and acceptance report.

