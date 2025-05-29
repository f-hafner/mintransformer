from __future__ import annotations
import torch
from .dataloading import get_batch
from .dataloading import load_data
from .models.bigram import BigramLanguageModel

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

@torch.no_grad() # more efficient b/c torch will not store intermediate variables (that are used for the backward pass)
def estimate_loss(data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Calculate the mean of train and validation losses -> less noise."""
    out = {}
    model.eval() # some layers behave differently when in training or evaluation mode
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data_dict[split], block_size=block_size, batch_size=batch_size, device=device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


data_dict, vocab_size, decode_fct = load_data()
model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        device=device)
m = model.to(device) # important for GPU

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iteration % eval_interval == 0:
        losses = estimate_loss(data_dict)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(data_dict["train"], block_size=block_size, batch_size=batch_size, device=device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_fct(m.generate(context, max_new_tokens=500)[0].tolist()))

