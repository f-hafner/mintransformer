"""Based on bigram model in zero-to-hero."""

from __future__ import annotations
import logging
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)


@dataclass
class BigramModelConfig:
    """Container for bigram model configuration."""

    vocab_size: int
    n_embd: int
    block_size: int
    n_head: int

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            msg = f"vocab_size must be positive, got {self.vocab_size}"
            raise ValueError(msg)
        if self.n_embd <= 0:
            msg = f"n_embd must be positive, got {self.n_embd}"
            raise ValueError(msg)
        if self.block_size <= 0:
            msg = f"block_size must be positive, got {self.block_size}"
            raise ValueError(msg)
        if self.n_head <= 0:
            msg = f"n_head must be positive, got {self.n_head}"
            raise ValueError(msg)
        if self.n_head > self.n_embd:
            msg = f"n_head ({self.n_head}) cannot exceed n_embd ({self.n_embd})"
            raise ValueError(msg)
        if self.n_embd % self.n_head != 0:
            msg = f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            raise ValueError(msg)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(num_heads)],
        )
        self.proj = nn.Linear(n_embd, n_embd)  # I think this is standard linear layer after attn head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        heads = [h(x) for h in self.heads]  # each head has shape (B, T, head_size)
        out = torch.cat(heads, dim=-1)  # (B, T, head_size * n_heads)
        return self.proj(out)


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.net = nn.Sequential(
            # paper, section 3.3: feedforward layers has n_head times the dimension of the embeddings
            nn.Linear(n_embd, n_head * n_embd),
            nn.ReLU(),
            # I understand here the projection is necessary because of the multiplication of channels (?)
            nn.Linear(n_head * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)  # note: b/c LayerNorm has 2 trainable params, we need
        # separate instances each time we use it
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        x = x + self.sa_heads(self.ln1(x))  # residual connection
        return x + self.ffwd(self.ln2(x))


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        _, n_targets, emb_dim = x.shape  # B, T, C
        k = self.key(x)  # B, T, head_size
        q = self.query(x)  # B, T, head_size
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * (emb_dim**-0.5)  # B, T, T
        wei = wei.masked_fill(self.tril[:n_targets, :n_targets] == 0, float("-inf"))
        wei = functional.softmax(wei, dim=-1)  # B, T, T

        v = self.value(x)  # B, T, head_size
        return wei @ v  # (B, T, T) * (B, T, head_size) = B, T, head_size


class BigramLanguageModel(nn.Module):
    """Super simple bigram model."""

    def __init__(self, cfg: BigramModelConfig):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(
            Block(cfg.n_embd, n_head=cfg.n_head, block_size=cfg.block_size),
            Block(cfg.n_embd, n_head=cfg.n_head, block_size=cfg.block_size),
            Block(cfg.n_embd, n_head=cfg.n_head, block_size=cfg.block_size),
            nn.LayerNorm(cfg.n_embd),
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(
        self,
        sources: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run forward pass."""
        logger.debug("Source device: %s", sources.device)
        if isinstance(targets, torch.Tensor):
            logger.debug("Targets device: %s", targets.device)
        else:
            logger.debug("Targets is None.")
        batch_size, n_targets = sources.shape  # B, T

        tok_emb = self.token_embedding_table(sources)  # (B, T, n_emb)
        pos_emb = self.position_embedding_table(
            torch.arange(n_targets, device=sources.device)
        )  # (T, n_emb) x = tok_emb + pos_emb  # (B, T, n_emb)
        logger.debug("pos_emb device: %s", pos_emb.device)

        x = tok_emb + pos_emb  # (B, T, n_emb)
        x = self.blocks(x)
        # note broadcasting: (B, T, n_emb) + (T, n_emb); the latter gets broadcast to (B, T, n_emb)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            batch_size, n_targets, n_classes = logits.shape  # B, T, C
            logits = logits.view(batch_size * n_targets, n_classes)
            targets = targets.view(batch_size * n_targets)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens."""
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens --
            # this is to prevent the positional embeddings from
            # running out of scope with longer inputs
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = functional.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
