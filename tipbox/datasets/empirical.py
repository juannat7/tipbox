"""Five empirical cases used by KEWS, from the pinned Grziwotz et al. archive.

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
class EmpiricalSpec:
    """Source labels and retrospective analysis settings, available offline.

    ``transition_time`` and ``trend_start`` are reference values used in the
    upstream plotting script and KEWS analysis; they are not event estimates
    inferred by this loader. ``example_window`` and ``example_step`` are sample
    counts from that analysis, rather than tuned classical-indicator defaults.
    Recovery intervals are inclusive (label, start, stop) triples.
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
    recovery_intervals: tuple[tuple[str, float, float], ...] = ()
    time_unit_notes: str = ""


_SPECS = {
    "microcosm": EmpiricalSpec(
        name="microcosm",
        display_name="Cyanobacterial microcosm",
        time_unit="day",
        observable="Light attenuation",
        observable_unit="m^-1",
        contributor="Egbert H. van Nes; Veraart et al. (2012), doi:10.1038/nature10723",
        transition_time=27.0,
        trend_start=0.0,
        example_window=200,
        example_step=5,
        recovery_intervals=(
            ("P1", 0.57092, 1.4539),
            ("P2", 4.5496, 5.4326),
            ("P3", 10.642, 11.525),
            ("P4", 14.454, 15.337),
            ("P5", 19.499, 20.382),
            ("P6", 24.502, 25.385),
        ),
    ),
    "voice": EmpiricalSpec(
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
    "mitochondria": EmpiricalSpec(
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
    "climate": EmpiricalSpec(
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
    "blackout": EmpiricalSpec(
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
    """Unmodified time and observations, plus source and sampling metadata.

    ``time`` has shape (n,), and ``observations`` has shape (n, n_observables).
    The five aggregated empirical cases each have one observed column. Nonfinite
    observations, duplicate times, and original row order are preserved. Metadata
    contains the specification fields, source DOI and checksum, archive member,
    full upstream terms, and sampling diagnostics. No class labels are assigned.
    """

    time: np.ndarray
    observations: np.ndarray
    metadata: dict


def list_empirical() -> tuple[str, ...]:
    """Names of the five supported empirical records; requires no network."""
    return tuple(sorted(_SPECS))


def get_empirical_info(name: str) -> EmpiricalSpec:
    """Inspect a case's labels, reference marker, and example controls offline."""
    if name not in _SPECS:
        raise ValueError(
            f"Unknown empirical case {name!r}; choose from {list_empirical()}"
        )
    return _SPECS[name]


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


def load_empirical(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
    archive_path: str | Path | None = None,
) -> EmpiricalRecord:
    """Load one pinned empirical record, preserving its raw sampling and gaps.

    Parameters
    ----------
    name : str
        A name from :func:`list_empirical`.
    cache_dir : path-like or None, default=None
        Cache root; defaults to $XDG_CACHE_HOME/tipbox or ~/.cache/tipbox.
        The archive lives in the zenodo_7379358 subdirectory.
    download : bool, default=True
        Download the 1 MB archive if absent. False permits cached/offline use only.
        Existing archives are always checked; corrupt caches raise ValueError and
        are not silently replaced. HTTP/network errors propagate to the caller.
    archive_path : path-like or None, default=None
        Use an already downloaded, unmodified Zenodo v1 ZIP. This overrides cache
        lookup and never downloads or extracts files. Size and SHA256 must match.

    Notes
    -----
    Only a named text member and its README are read; archive files are never
    extracted. No rows are removed, sorted, interpolated, detrended, or normalized.
    Upstream terms require acknowledgement, restrict monitoring data to nonprofit
    use, and require contact with data administrators before publication or
    presentations; see metadata["terms"] and the source record. These terms are
    separate from TipBox's MIT software license. Reference markers are
    retrospective annotations, not predictions made by this loader.
    """
    spec = get_empirical_info(name)
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
