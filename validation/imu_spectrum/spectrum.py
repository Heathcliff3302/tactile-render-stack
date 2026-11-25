"""P2.4 spectral estimation: demean, Hann window, Welch PSD, band powers.

Pure numpy, no scipy, so it runs in the Phase 2 environment as is.

Conventions, stated once because every number downstream inherits them:

* PSD is one-sided, density-scaled, in (m/s^2)^2 / Hz. With density scaling
  and the Hann window-power correction, integrating the PSD over frequency
  returns the signal variance (Parseval), which a unit test checks.
* Segment length is set in **seconds** (``DEFAULT_SEGMENT_S``), not samples,
  so the frequency resolution is the same 2 Hz at any ODR. A 3840 Hz and a
  7690 Hz recording therefore produce comparable band powers.
* Band power is the PSD integrated over the bins whose centre frequency lies
  in ``[lower_hz, upper_hz)``. A band with no bin, or above Nyquist, is
  ``None``.
* Three axes are kept separately and also summed ("total"). Summing powers is
  orientation independent; it is not the DFT321 single-axis synthesis, which
  is a perceptual step and belongs to layer 4b, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import BANDS, available

DEFAULT_SEGMENT_S = 0.5
DEFAULT_OVERLAP = 0.5
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Psd:
    frequency_hz: np.ndarray
    density: np.ndarray            # (m/s^2)^2 / Hz
    sample_rate_hz: float
    segment_samples: int
    averages: int

    @property
    def resolution_hz(self) -> float:
        return self.sample_rate_hz / self.segment_samples


def welch_psd(signal: np.ndarray, sample_rate_hz: float, *, segment_s: float = DEFAULT_SEGMENT_S,
              overlap: float = DEFAULT_OVERLAP) -> Psd:
    """One-sided Welch PSD of a demeaned signal with a Hann window."""
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    n = int(round(segment_s * sample_rate_hz))
    if n < 8:
        raise ValueError("segment too short for a spectrum")
    if x.size < n:
        raise ValueError(f"signal has {x.size} samples, fewer than one {n}-sample Welch segment")
    step = max(1, int(round(n * (1.0 - overlap))))
    window = np.hanning(n + 2)[1:-1]  # symmetric Hann without the zero end points
    scale = 1.0 / (sample_rate_hz * np.sum(window ** 2))
    starts = range(0, x.size - n + 1, step)
    accumulator = np.zeros(n // 2 + 1)
    count = 0
    for start in starts:
        piece = x[start:start + n]
        piece = (piece - piece.mean()) * window
        spectrum = np.abs(np.fft.rfft(piece)) ** 2 * scale
        accumulator += spectrum
        count += 1
    density = accumulator / count
    # One-sided: double every bin except DC and (for even n) Nyquist.
    if n % 2 == 0:
        density[1:-1] *= 2.0
    else:
        density[1:] *= 2.0
    frequency = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    return Psd(frequency, density, sample_rate_hz, n, count)


def band_powers(psd: Psd) -> list[float | None]:
    """Integrated power per shared band, ``None`` where the band is unmeasurable."""
    df = psd.resolution_hz
    powers: list[float | None] = []
    for band in BANDS:
        if not available(band, psd.sample_rate_hz):
            powers.append(None)
            continue
        mask = (psd.frequency_hz >= band.lower_hz) & (psd.frequency_hz < band.upper_hz)
        powers.append(float(psd.density[mask].sum() * df) if mask.any() else None)
    return powers


def analysis_range(sample_rate_hz: float) -> tuple[float, float]:
    """Frequency range covered by the available bands at this sample rate."""
    usable = [band for band in BANDS if available(band, sample_rate_hz)]
    if not usable:
        raise ValueError(f"no band lies below Nyquist at {sample_rate_hz} Hz")
    return usable[0].lower_hz, usable[-1].upper_hz


def peak_frequency(frequency_hz: np.ndarray, density: np.ndarray, low_hz: float,
                   high_hz: float) -> float | None:
    """Frequency of the largest positive density value in ``[low, high)``."""
    mask = (frequency_hz >= low_hz) & (frequency_hz < high_hz)
    if not mask.any():
        return None
    values = np.where(mask, density, -np.inf)
    index = int(np.argmax(values))
    return float(frequency_hz[index]) if values[index] > 0 else None


def spectral_centroid(frequency_hz: np.ndarray, density: np.ndarray, low_hz: float,
                      high_hz: float) -> float | None:
    """Power-weighted mean frequency of the positive part of ``density``.

    Sandpaper is random roughness, not a grating, so its spectrum may be a
    broad hump without one sharp peak. The centroid still scales as v/lambda
    under the same kinematics, so it is reported beside the peak.
    """
    mask = (frequency_hz >= low_hz) & (frequency_hz < high_hz)
    positive = np.clip(np.where(mask, density, 0.0), 0.0, None)
    total = positive.sum()
    if total <= 0:
        return None
    return float((frequency_hz * positive).sum() / total)


def axis_psds(accel_mps2: dict[str, np.ndarray], sample_rate_hz: float, **kwargs) -> dict[str, Psd]:
    """Per-axis PSDs plus their sum under the key ``total``."""
    psds = {axis: welch_psd(accel_mps2[axis], sample_rate_hz, **kwargs) for axis in AXES}
    first = psds[AXES[0]]
    psds["total"] = Psd(first.frequency_hz, sum(psds[axis].density for axis in AXES),
                        sample_rate_hz, first.segment_samples, first.averages)
    return psds
