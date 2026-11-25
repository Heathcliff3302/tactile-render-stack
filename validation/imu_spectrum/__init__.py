"""P2.4 IMU texture-spectrum analysis (analysis side only; no data is bundled).

    cd Phase_2
    ./.venv/Scripts/python.exe -m validation.imu_spectrum --session <session.json> --out <dir>

``bands`` is the single band definition P2.7.1 must import. ``ingest`` loads
and rejects recordings. ``spectrum`` and ``analysis`` compute. Synthetic
signals live only in ``validation/tests/test_imu_spectrum.py``; a session must
declare ``data_origin = "measured"`` to reach the command-line report.
"""

from .bands import BANDS, BAND_SCHEME_VERSION, definition
from .ingest import IngestPolicy, RecordingRejected, Session, load_session
from .analysis import analyse, write_report

__all__ = ["BANDS", "BAND_SCHEME_VERSION", "IngestPolicy", "RecordingRejected", "Session",
           "analyse", "definition", "load_session", "write_report"]
