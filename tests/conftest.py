"""Test fixtures and sample data for dataloading tests."""

from __future__ import annotations
import pytest
from mintransformer.dataloading.datasets import write_to_arrow_stream


@pytest.fixture(scope="session")
def arrow_file(sample_sequences, tmp_path_factory) -> tuple[str, dict[str, list[list[int]]]]:
    """Fixture creating a temporary arrow file with sample data."""
    data_dict = sample_sequences
    fn = tmp_path_factory.mktemp("data") / "sequence_file.arrow"
    fn = str(fn)
    write_to_arrow_stream(data=data_dict, filename=fn)
    return fn, data_dict


@pytest.fixture(scope="session")
def sample_sequences() -> dict[str, list[list[int]]]:
    """Create sample integer sequence data for testing."""
    # Example: simple sequences where y is x shifted by 1
    x_sequences = [
        [3, 5, 7, 8, 5, 10],
        [1, 2, 3, 4, 5, 6],
        [10, 9, 8, 7, 6, 5],
        [2, 4, 6, 8, 10, 12],
        [15, 14, 13, 12, 11, 10],
        [20, 22, 24, 26, 28, 30],
    ]

    y_sequences = [
        [5, 7, 8, 5, 10, 15],
        [2, 3, 4, 5, 6, 7],
        [9, 8, 7, 6, 5, 4],
        [4, 6, 8, 10, 12, 14],
        [14, 13, 12, 11, 10, 9],
        [22, 24, 26, 28, 30, 32],
    ]

    return {"x": x_sequences, "y": y_sequences}
