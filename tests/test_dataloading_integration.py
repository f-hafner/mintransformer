"""Integration tests for dataloading components."""

from __future__ import annotations
import torch
from mintransformer.dataloading.datasets import ArrowDataset
from mintransformer.dataloading.datasets import ArrowReader
from mintransformer.dataloading.datasets import CustomDataset


def test_arrow_dataset_integration(arrow_file) -> None:
    """Test ArrowDataset reads data correctly and yields expected x,y pairs."""
    filename, expected_data = arrow_file
    # Load dataset using ArrowReader
    dataset_kwargs = ArrowReader().read(filename=filename, in_memory=True)
    arrow_dataset = ArrowDataset(**dataset_kwargs)

    # Collect all samples
    samples = list(arrow_dataset)

    # Verify we got the expected number of samples
    assert len(samples) == len(expected_data["x"])

    # Test first record specifically
    first_sample = samples[0]
    expected_first_x = torch.tensor(expected_data["x"][0])
    expected_first_y = torch.tensor(expected_data["y"][0])

    assert "x" in first_sample
    assert "y" in first_sample
    assert torch.equal(first_sample["x"], expected_first_x)
    assert torch.equal(first_sample["y"], expected_first_y)

    # Verify all samples have correct structure and data
    for i, sample in enumerate(samples):
        expected_x = torch.tensor(expected_data["x"][i])
        expected_y = torch.tensor(expected_data["y"][i])

        assert torch.equal(sample["x"], expected_x)
        assert torch.equal(sample["y"], expected_y)


def test_custom_dataset_integration(arrow_file) -> None:
    """Test CustomDataset processes ArrowDataset correctly."""
    filename, expected_data = arrow_file

    # Load arrow dataset
    dataset_kwargs = ArrowReader().read(filename=filename, in_memory=True)
    arrow_dataset = ArrowDataset(**dataset_kwargs)

    # Create CustomDataset
    custom_dataset = CustomDataset(arrow_dataset)

    # Verify dataset length
    assert len(custom_dataset) == len(expected_data["x"])

    # Verify all items
    for i in range(len(custom_dataset)):
        item = custom_dataset[i]
        expected_x = torch.tensor(expected_data["x"][i])
        expected_y = torch.tensor(expected_data["y"][i])

        assert len(item) == len(expected_data)
        assert torch.equal(item[0], expected_x)
        assert torch.equal(item[1], expected_y)
