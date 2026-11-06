"""Full-precision cyanobacterial chemostat observations from their original deposit."""

from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from ._cache import pinned_file
from .empirical import EmpiricalRecord, _sampling

SOURCE_URL = "https://lifesciences.datastations.nl/api/access/datafile/14452"
SOURCE_NAME = "Correct data both chemostats.ods"
SOURCE_SIZE = 897_788
SOURCE_SHA256 = "cec2680a8af7cc13e8afe31f92b9315d9c481b23523d73f9c47b552f75132f3e"
_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"


class Microcosm:
    """Cyanobacterial chemostat observations, M1 and M2 (Veraart et al., 2012)."""

    def load(
        self,
        *,
        cache_dir: str | Path | None = None,
        download: bool = True,
        source_path: str | Path | None = None,
    ) -> EmpiricalRecord:
        """Load both chemostats, preserving source times and missing values.

        cache_dir selects the cache; download=False requires a cached file.
        source_path reads a local, unmodified ODS instead. Files are verified
        against the published source checksum. Observations have columns M1, M2.
        """
        path = pinned_file(
            SOURCE_URL,
            Path("dans_ztg_93aw") / SOURCE_NAME,
            size=SOURCE_SIZE,
            checksum=SOURCE_SHA256,
            cache_dir=cache_dir,
            local_path=source_path,
            download=download,
        )
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("content.xml"))
        sheet = root.find(f".//{{{_TABLE}}}table")
        rows = []
        for row in sheet.findall(f"{{{_TABLE}}}table-row"):
            cells = list(row)
            if not cells or cells[0].get(f"{{{_OFFICE}}}value-type") != "float":
                continue
            values = []
            for cell in cells:
                value = float(cell.get(f"{{{_OFFICE}}}value", "nan"))
                repeat = int(cell.get(f"{{{_TABLE}}}number-columns-repeated", "1"))
                values.extend([value] * min(repeat, 3 - len(values)))
                if len(values) == 3:
                    break
            rows.append(values)
        data = np.asarray(rows)
        metadata = {
            "name": "microcosm",
            "display_name": "Cyanobacterial microcosms",
            "observable": "Light attenuation",
            "observable_names": ("M1", "M2"),
            "observable_unit": "m^-1",
            "time_unit": "day",
            "source_doi": "10.17026/dans-ztg-93aw",
            "source_url": "https://doi.org/10.17026/dans-ztg-93aw",
            "paper_doi": "10.1038/nature10723",
            "citation": "Veraart et al. (2012), Nature 481, 357–359",
            "source_file": SOURCE_NAME,
            "source_path": str(path.resolve()),
            "source_sha256": SOURCE_SHA256,
            "datafile_id": 14452,
            "license": "CC-BY-NC-4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "preprocessing": "none",
            "sampling_by_observable": {
                name: _sampling(data[:, 0], data[:, index + 1])
                for index, name in enumerate(("M1", "M2"))
            },
        }
        return EmpiricalRecord(
            time=data[:, 0], observations=data[:, 1:3], metadata=metadata
        )
