"""The one frequency-band definition shared by P2.4 and layer 4a (P2.7.1).

P2.7.1 requires that the vibration-spectrum channel of ``TactileTarget`` and
the P2.4 analysis use the *same* band edges, not two hand-written copies. This
module is that single definition; P2.7.1 must import it rather than restate it.

Bands are IEC 61260 / ISO 266 one-third-octave bands, base 10: nominal centres
from 25 Hz to 3150 Hz, exact centre ``1000 * 10**(n/10)``, edges at
``centre * 10**(+-1/20)``. A standard was chosen over a hand-picked set so the
boundaries carry no tuning: nobody chose them to make a texture peak land
anywhere in particular.

Why 25 Hz to 3150 Hz. The upper edge of the 3150 Hz band is 3536 Hz, below the
3845 Hz Nyquist of the 7690 Hz ODR declared in the P2.4 plan. The lower end
keeps at least two Welch bins in the narrowest band at the default resolution
(``spectrum.DEFAULT_SEGMENT_S`` = 0.5 s gives 2 Hz bins, about three in the 5.8 Hz-wide 25 Hz band).

A band is *available* for a recording only when its upper edge is at or below
that recording's Nyquist frequency. Otherwise its power is ``None`` — not
zero. At a 3840 Hz ODR (the I2C fallback in the plan, Nyquist 1920 Hz) the
top bands are therefore unavailable, which is the truthful answer.
"""

from __future__ import annotations

from dataclasses import dataclass

BAND_SCHEME = "iec61260_one_third_octave_base10"
BAND_SCHEME_VERSION = "p2.4-bands/v1"

#: Exponents n in centre = 1000 * 10**(n/10). n = -16 is 25 Hz, n = 5 is 3150 Hz.
_EXPONENTS = range(-16, 6)

#: ISO 266 nominal labels for those centres, in the same order.
_NOMINAL_HZ = (25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
               630, 800, 1000, 1250, 1600, 2000, 2500, 3150)


@dataclass(frozen=True)
class Band:
    index: int
    nominal_hz: float
    centre_hz: float
    lower_hz: float
    upper_hz: float

    @property
    def label(self) -> str:
        value = self.nominal_hz
        return f"{int(value)}Hz" if float(value).is_integer() else f"{value}Hz"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "nominal_hz": self.nominal_hz,
            "centre_hz": self.centre_hz,
            "lower_hz": self.lower_hz,
            "upper_hz": self.upper_hz,
        }


def _build() -> tuple[Band, ...]:
    bands = []
    for index, (exponent, nominal) in enumerate(zip(_EXPONENTS, _NOMINAL_HZ, strict=True)):
        centre = 1000.0 * 10.0 ** (exponent / 10.0)
        bands.append(Band(
            index=index,
            nominal_hz=float(nominal),
            centre_hz=centre,
            lower_hz=centre * 10.0 ** (-1.0 / 20.0),
            upper_hz=centre * 10.0 ** (1.0 / 20.0),
        ))
    return tuple(bands)


BANDS: tuple[Band, ...] = _build()


def available(band: Band, sample_rate_hz: float) -> bool:
    """A band is measurable only if it lies entirely below Nyquist."""
    return band.upper_hz <= sample_rate_hz / 2.0


def definition() -> dict:
    """Serialisable band table, stamped into every analysis report."""
    return {
        "scheme": BAND_SCHEME,
        "version": BAND_SCHEME_VERSION,
        "availability_rule": "upper_hz <= sample_rate_hz / 2, else power is null",
        "bands": [band.to_dict() for band in BANDS],
    }
