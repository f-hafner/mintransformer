import torch
from mintransformer.dataloading import create_splits
from mintransformer.dataloading import reshape_to_batches


def test_reshape_to_batches():
    """Test that reshape works with offset."""
    n_batches = 12
    block_size = 8
    offset = 3
    a = torch.randint(0, 400, (n_batches * block_size + offset,))
    expected = torch.reshape(a[:-offset], (n_batches, block_size))
    output = reshape_to_batches(a, block_size)
    assert torch.all(output == expected)


def test_create_splits():
    """Test train/test split and creation of trainin data."""
    n_samples = 500
    train_frac = 0.9
    n_train = int(train_frac * n_samples)
    a = torch.randint(0, 400, (n_samples,))
    result = create_splits(a, train_frac)
    result_keys = list(result.keys())
    expected_keys = ["x_train", "y_train", "x_test", "y_test"]
    assert len(result_keys) == len(expected_keys)
    assert all(x in result for x in expected_keys)
    assert result["x_train"][0] == a[0]
    assert result["y_train"][0] == a[1]

    assert result["x_test"][0] == a[n_train]
    assert result["y_test"][0] == a[n_train + 1]
