from validation.isaac_parity import THRESHOLDS, compare, load_kernel


def test_kernel_against_itself_passes_every_gate(k1_run_directory, k2_run_directory):
    for directory, spec in ((k1_run_directory, "k1_single_probe_cpu"), (k2_run_directory, "k2_force_cpu")):
        frames = load_kernel(directory)
        assert compare(frames, frames, spec)["all_gates_pass"]


def test_one_step_contact_lag_is_caught(k1_run_directory):
    """The B3 finding: a point one slide step late must fail the 0.1 mm gate."""
    frames = load_kernel(k1_run_directory)
    shifted = [dict(f) for f in frames]
    for f in shifted:
        if f["phase"] == "lateral_slide" and f["contact_points"]:
            f["contact_points"] = [
                {**p, "position_world_m": [p["position_world_m"][0] - 0.05 / 60, *p["position_world_m"][1:]]}
                for p in f["contact_points"]
            ]
    result = compare(frames, shifted, "k1_single_probe_cpu")
    assert not result["gates"]["contact_point"]["pass"]
    assert result["gates"]["contact_point"]["max_m"] > THRESHOLDS["contact_point_max_m"]
