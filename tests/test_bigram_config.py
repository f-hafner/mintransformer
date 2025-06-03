"""Tests for BigramModelConfig validation."""

import pytest
from mintransformer.models.bigram import BigramModelConfig


class TestBigramModelConfigValidation:
    """Test parameter validation in BigramModelConfig."""

    def test_valid_config(self):
        """Test that valid parameters pass validation."""
        vocab_size = 100
        n_embd = 32
        block_size = 8
        n_head = 4
        config = BigramModelConfig(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head)
        assert config.vocab_size == vocab_size
        assert config.n_embd == n_embd
        assert config.block_size == block_size
        assert config.n_head == n_head

    def test_negative_vocab_size(self):
        """Test that negative vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive, got -10"):
            BigramModelConfig(vocab_size=-10, n_embd=32, block_size=8, n_head=4)

    def test_zero_vocab_size(self):
        """Test that zero vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive, got 0"):
            BigramModelConfig(vocab_size=0, n_embd=32, block_size=8, n_head=4)

    def test_negative_n_embd(self):
        """Test that negative n_embd raises ValueError."""
        with pytest.raises(ValueError, match="n_embd must be positive, got -5"):
            BigramModelConfig(vocab_size=100, n_embd=-5, block_size=8, n_head=4)

    def test_zero_n_embd(self):
        """Test that zero n_embd raises ValueError."""
        with pytest.raises(ValueError, match="n_embd must be positive, got 0"):
            BigramModelConfig(vocab_size=100, n_embd=0, block_size=8, n_head=4)

    def test_negative_block_size(self):
        """Test that negative block_size raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be positive, got -3"):
            BigramModelConfig(vocab_size=100, n_embd=32, block_size=-3, n_head=4)

    def test_zero_block_size(self):
        """Test that zero block_size raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be positive, got 0"):
            BigramModelConfig(vocab_size=100, n_embd=32, block_size=0, n_head=4)

    def test_negative_n_head(self):
        """Test that negative n_head raises ValueError."""
        with pytest.raises(ValueError, match="n_head must be positive, got -2"):
            BigramModelConfig(vocab_size=100, n_embd=32, block_size=8, n_head=-2)

    def test_zero_n_head(self):
        """Test that zero n_head raises ValueError."""
        with pytest.raises(ValueError, match="n_head must be positive, got 0"):
            BigramModelConfig(vocab_size=100, n_embd=32, block_size=8, n_head=0)

    def test_n_embd_not_divisible_by_n_head(self):
        """Test that n_embd not divisible by n_head raises ValueError."""
        with pytest.raises(ValueError, match="n_embd \\(33\\) must be divisible by n_head \\(4\\)"):
            BigramModelConfig(vocab_size=100, n_embd=33, block_size=8, n_head=4)

    def test_n_head_exceeds_n_embd(self):
        """Test that n_head > n_embd raises ValueError."""
        with pytest.raises(ValueError, match="n_head \\(10\\) cannot exceed n_embd \\(8\\)"):
            BigramModelConfig(vocab_size=100, n_embd=8, block_size=8, n_head=10)

    def test_edge_case_n_head_equals_n_embd(self):
        """Test that n_head == n_embd is valid."""
        config = BigramModelConfig(vocab_size=100, n_embd=8, block_size=8, n_head=8)
        assert config.n_head == config.n_embd
