import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size    = 64
block_size    = 256
max_iters     = 5000
eval_interval = 500
learning_rate = 3e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)
stoi       = { ch:i for i, ch in enumerate(chars)}
itos       = { i:ch for i, ch in enumerate(chars)}
encode     = lambda x: [ stoi[ch] for ch in x ]
decode     = lambda x: ''.join([ itos[ch] for ch in x ])

data       = torch.tensor(encode(text), dtype = torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size, ))
    x    = torch.stack([data[i   : i+block_size]   for i in ix])
    y    = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection in Sequential container. proj going back to residual way.
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block, communication followed by computation."""

    def __init__(self, n_embd, n_head):
        # n_embd:embedding dimension, n_head: the number of heads we'd like.
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        return x

class LayerNorm:
    """affine will be true which means using gamma and beta during training.
    track_running_stats will be true, keep tracking running mean and variance.
    device by default is cpu(but i hope is gpu.).
    datatype float32.
    """
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps      = eps
        # self.momentum = momentum
        # self.training = True

        # parameters (trained with backprop.)
        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)

        # buffers (trained with a running momentum update)
        # self.running_mean = torch.zeros(dim)
        # self.running_var  = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass.
        # if self.training:
        #     # if x.ndim == 2:
        #     #     dim = 0
        #     # elif x.ndim == 3:
        #     #     dim = (0, 1)
        #     xmean = x.mean(1, keepdim = True) # batch mean
        #     xvar  =  x.var(1, keepdim = True)  # batch variance
        # else:
        xmean = self.running_mean
        xvar  = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalized to unit variance
        self.out = self.gamma * xhat + self.beta
        # if self.training:
        #     with torch.no_grad():
        #         self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        #         self.running_var  = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            nn.LayerNorm(n_embd),
        )
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # short for language model head.

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x       = tok_emb + pos_emb
        # x       = self.sa_heads(x)
        # x       = self.ffwd(x)
        x       = self.blocks(x)
        logits  = self.lm_head(x)

        # logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # # get the predictions
            # logits, loss = self(idx)
            # # focus only on the last time step
            # logits = logits[:, -1, :] # becomes (B, C)
            # # apply softmax to get probabilities
            # probs = F.softmax(logits, dim=-1) # (B, C)
            # # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # # append sampled index to the running sequence
            # idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))