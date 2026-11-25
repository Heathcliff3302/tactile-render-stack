"""Command-line entry: measured session in, report.json and band_table.csv out."""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import analyse, write_report
from .ingest import IngestPolicy, RecordingRejected, load_session
from .spectrum import DEFAULT_SEGMENT_S


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P2.4 IMU texture-spectrum analysis")
    parser.add_argument("--session", required=True, help="session manifest (p2.4-imu-session/v1)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--segment-s", type=float, default=DEFAULT_SEGMENT_S,
                        help="Welch segment length in seconds (sets the 1/segment_s Hz resolution)")
    parser.add_argument("--max-interval-factor", type=float, default=IngestPolicy.max_interval_factor)
    parser.add_argument("--odr-tolerance-rel", type=float, default=IngestPolicy.odr_tolerance_rel)
    args = parser.parse_args(argv)
    policy = IngestPolicy(max_interval_factor=args.max_interval_factor,
                          odr_tolerance_rel=args.odr_tolerance_rel)
    try:
        session = load_session(args.session, policy=policy, require_measured=True)
    except RecordingRejected as error:
        print(f"REJECTED: {error}", file=sys.stderr)
        return 2
    paths = write_report(analyse(session, segment_s=args.segment_s), args.out)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
