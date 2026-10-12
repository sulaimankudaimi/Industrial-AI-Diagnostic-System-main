"""Smoke tests for the diagnostic package skeleton."""

import importlib.util

import pytest

import diagnostic


def test_package_imports() -> None:
    """The package imports and exposes the expected version."""
    assert diagnostic.__version__ == "0.1.0"


@pytest.mark.ml
@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="tensorflow is not installed",
)
def test_tensorflow_importable() -> None:
    """TensorFlow is importable when the optional ml extra is installed."""
    import tensorflow as tf

    assert tf.__version__
