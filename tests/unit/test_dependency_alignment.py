"""Guards the version contract between the app image and the PyFlink image.

Model artifacts are joblib pickles written by training and read by both images.
Those pickles do not survive a numpy major-version boundary: MLPRegressor embeds
an MT19937 BitGenerator and ndarrays reference numpy._core, neither of which
numpy 1 can read from a numpy 2 pickle. A drift between the two requirement
files therefore surfaces as a Flink job that dies at model load, which is a long
way from the file that caused it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

APP_REQUIREMENTS = Path("requirements.txt")
FLINK_REQUIREMENTS = Path("infra/flink/requirements.txt")

# Packages whose versions must match, because their objects are inside the pickles
# or they govern how those objects are reconstructed.
PICKLE_CRITICAL = ("numpy", "scikit-learn", "scipy", "joblib")

# apache-flink 1.19.1 pins numpy<1.25. Raising this ceiling requires upgrading
# Flink first, so the bound is asserted rather than left implicit.
PYFLINK_NUMPY_CEILING = (1, 25)

_PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s;#]+)")


def _parse_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.match(line)
        if match:
            pins[match.group("name").lower()] = match.group("version")
    return pins


@pytest.fixture(scope="module")
def app_pins() -> dict[str, str]:
    return _parse_pins(APP_REQUIREMENTS)


@pytest.fixture(scope="module")
def flink_pins() -> dict[str, str]:
    return _parse_pins(FLINK_REQUIREMENTS)


@pytest.mark.parametrize("package", PICKLE_CRITICAL)
def test_pickle_critical_versions_match_across_images(package, app_pins, flink_pins):
    app_version = app_pins.get(package)
    flink_version = flink_pins.get(package)

    assert app_version is not None, f"{package} missing from {APP_REQUIREMENTS}"
    if flink_version is None:
        # scipy arrives transitively in the Flink image via scikit-learn, so an
        # explicit pin is optional there. Matching scikit-learn is what fixes it.
        pytest.skip(f"{package} is not pinned directly in {FLINK_REQUIREMENTS}")

    assert app_version == flink_version, (
        f"{package} differs: {APP_REQUIREMENTS} has {app_version}, "
        f"{FLINK_REQUIREMENTS} has {flink_version}. Model pickles cross both images."
    )


def test_numpy_stays_under_the_pyflink_ceiling(app_pins):
    version = app_pins["numpy"]
    parts = tuple(int(p) for p in version.split(".")[:2])
    assert parts < PYFLINK_NUMPY_CEILING, (
        f"numpy {version} exceeds the apache-flink 1.19.1 ceiling "
        f"{PYFLINK_NUMPY_CEILING[0]}.{PYFLINK_NUMPY_CEILING[1]}. "
        "Upgrade Flink before raising numpy, or the streaming job cannot load model artifacts."
    )


def test_scikit_learn_supports_the_flink_image_python(flink_pins):
    # The flink:1.19.1 base image is Ubuntu 22.04, whose python3 is 3.10.
    # scikit-learn 1.8+ requires Python 3.11 and cannot be installed there.
    major, minor = (int(p) for p in flink_pins["scikit-learn"].split(".")[:2])
    assert (major, minor) <= (1, 7), (
        f"scikit-learn {flink_pins['scikit-learn']} requires Python 3.11, but the "
        "flink:1.19.1 base image provides Python 3.10."
    )
