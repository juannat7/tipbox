"""Empirical observations from the pinned Grziwotz et al. archive.

Data are downloaded on demand into a user cache, never bundled with TipBox.
The archive's data-use terms apply separately from the package's MIT license.
"""

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from ._cache import pinned_file

ARCHIVE_URL = "https://zenodo.org/records/7379358/files/DEV_Bifurcation-main.zip"
ARCHIVE_NAME = "DEV_Bifurcation-main.zip"
ARCHIVE_SIZE = 1_016_323
ARCHIVE_SHA256 = "5e5fb299890014f6053e98c3e63b93a21dd9c7bff9212391c9cfd2e7d5dc9275"
SOURCE_DOI = "10.5281/zenodo.7379358"
PAPER_DOI = "10.1126/sciadv.abq4558"
ARCHIVE_ROOT = "DEV_Bifurcation-main"
EMPIRICAL_SUBDIR = f"{ARCHIVE_ROOT}/figure_2_empirical_examples"
TERMS_MEMBER = f"{ARCHIVE_ROOT}/README.txt"


@dataclass(frozen=True)
class _EmpiricalSpec:
    """Source labels and retrospective analysis settings, available offline.

    ``transition_time`` and ``trend_start`` are reference values used in the
    upstream plotting script and KEWS analysis; they are not event estimates
    inferred by this loader. ``example_window`` and ``example_step`` are sample
    counts from that analysis, rather than tuned indicator defaults.
    ``time_unit_notes`` records source inconsistencies; an unresolved unit must
    be clarified before assigning physical durations to its raw time coordinates.
    """

    name: str
    display_name: str
    time_unit: str
    observable: str
    observable_unit: str
    contributor: str
    transition_time: float
    trend_start: float
    example_window: int
    example_step: int
    time_column: int = 0
    value_column: int = 1
    time_unit_notes: str = ""


_SPECS = {
    "voice": _EmpiricalSpec(
        name="voice",
        display_name="Voice onset",
        time_unit="s",
        observable="Subglottal pressure",
        observable_unit="kPa",
        contributor="Isao T. Tokuda",
        transition_time=0.49,
        trend_start=0.3,
        example_window=200,
        example_step=50,
    ),
    "mitochondria": _EmpiricalSpec(
        name="mitochondria",
        display_name="Cellular ATP",
        time_unit="unresolved",
        observable="Relative fluorescence ratio",
        observable_unit="relative ratio",
        contributor="Markus Schwarzländer",
        transition_time=4.7,
        trend_start=3.2,
        example_window=125,
        example_step=1,
        time_unit_notes=(
            "The archive script and Grziwotz et al. (2023) Fig.3G-H label seconds, "
            "but the cellular-energy methods report minutes and a 40-second "
            "acquisition interval. The raw median increment is about 0.011111; "
            "no physical scale is assumed or conversion applied. Original "
            "experiment: Wagner et al. (2019), doi:10.1111/nph.16093."
        ),
    ),
    "climate": _EmpiricalSpec(
        name="climate",
        display_name="Greenhouse Earth climate",
        time_unit="million years, signed age increasing toward present",
        observable="CaCO3",
        observable_unit="%",
        contributor="World Data Center for Paleoclimatology; Dakos et al. (2008)",
        transition_time=-34.0,
        trend_start=-38.0,
        example_window=150,
        example_step=5,
        time_column=1,
        value_column=0,
    ),
    "blackout": _EmpiricalSpec(
        name="blackout",
        display_name="Electrical blackout",
        time_unit="s",
        observable="Frequency",
        observable_unit="Hz",
        contributor="Bonneville Power Administration",
        transition_time=550.0,
        trend_start=200.0,
        example_window=1000,
        example_step=50,
    ),
}


@dataclass(frozen=True)
class EmpiricalRecord:
    """Time (T,), observations (T,D), and source/sampling metadata."""

    time: np.ndarray
    observations: np.ndarray
    metadata: dict


def _sampling(time, observations):
    delta = np.diff(time)
    finite_delta = delta[np.isfinite(delta)]
    median = float(np.median(finite_delta))
    return {
        "n_observations": int(time.size),
        "n_missing_observations": int((~np.isfinite(observations)).sum()),
        "n_missing_times": int((~np.isfinite(time)).sum()),
        "n_duplicate_adjacent_times": int((delta == 0).sum()),
        "n_backward_steps": int((delta < 0).sum()),
        "dt_min": float(finite_delta.min()),
        "dt_median": median,
        "dt_max": float(finite_delta.max()),
        "is_regular": bool(
            np.isfinite(delta).all()
            and (delta > 0).all()
            and np.allclose(delta, median, rtol=1e-5, atol=0)
        ),
        "regularity_rtol": 1e-5,
    }


def _load_record(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
    archive_path: str | Path | None = None,
) -> EmpiricalRecord:
    """Read one record from the verified source archive."""
    spec = _SPECS[name]
    path = pinned_file(
        ARCHIVE_URL,
        Path("zenodo_7379358") / ARCHIVE_NAME,
        size=ARCHIVE_SIZE,
        checksum=ARCHIVE_SHA256,
        cache_dir=cache_dir,
        local_path=archive_path,
        download=download,
    )
    member = f"{EMPIRICAL_SUBDIR}/{name}.txt"
    with ZipFile(path) as archive:
        raw = np.genfromtxt(
            BytesIO(archive.read(member)),
            dtype=float,
            missing_values="NA",
            filling_values=np.nan,
        )
        terms = archive.read(TERMS_MEMBER).decode("utf-8")
    time = raw[:, spec.time_column]
    observations = raw[:, spec.value_column, None]
    metadata = {
        **asdict(spec),
        "source_doi": SOURCE_DOI,
        "source_url": "https://zenodo.org/records/7379358",
        "paper_doi": PAPER_DOI,
        "citation": "Grziwotz et al. (2023), Science Advances 9, eabq4558",
        "archive_url": ARCHIVE_URL,
        "archive_path": str(path.resolve()),
        "archive_sha256": ARCHIVE_SHA256,
        "source_member": member,
        "terms_member": TERMS_MEMBER,
        "terms": terms,
        "sampling": _sampling(time, observations),
        "preprocessing": "none",
    }
    return EmpiricalRecord(time=time, observations=observations, metadata=metadata)


class _ArchivedCase:
    _name: str

    def load(
        self,
        *,
        cache_dir: str | Path | None = None,
        download: bool = True,
        archive_path: str | Path | None = None,
    ) -> EmpiricalRecord:
        """Load observations; cache_dir selects the cache, download=False uses it offline.

        archive_path reads a local, unmodified source ZIP instead of the cache.
        """
        return _load_record(
            self._name,
            cache_dir=cache_dir,
            download=download,
            archive_path=archive_path,
        )


class VoiceOnset(_ArchivedCase):
    """Subglottal pressure during voice onset, from the Grziwotz et al. archive."""

    _name = "voice"


class CellularATP(_ArchivedCase):
    """Cellular fluorescence ratio; physical time units remain unresolved."""

    _name = "mitochondria"


class Paleoclimate(_ArchivedCase):
    """Greenhouse Earth CaCO3 record, retaining signed geological ages."""

    _name = "climate"


class ElectricalBlackout(_ArchivedCase):
    """Electrical frequency around a blackout, from the Grziwotz et al. archive."""

    _name = "blackout"
