import torch
import torch.nn as nn

# Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size, block_size = 32, 8
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
vocab_size = 65
n_embed = 32

# Basic setup
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


# split training and evaluation data
torch.manual_seed(1337)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
eval_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else eval_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    m.train()
    return out


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = nn.functional.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


m = BigramLM()
m = m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
idx = torch.zeros((1, 1), dtype=torch.long, device=device)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: training loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

generation = m.generate(idx, max_new_tokens=500)[0].tolist()

# mathematical trick of self-attention
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# single head to perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
k, q = key(x), query(x)
wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

# xbow = torch.zeros((B, T, C))  # bow->bag of words
tril = torch.tril((torch.ones(T, T)))
# wei = torch.zeros((T, T))
# wei = wei / torch.sum(wei, 1, keepdim=True)
wei = wei.masked_fill(tril == 0, float('-inf'))
# wei = nn.functional.softmax(wei, dim=-1)
wei = nn.functional.softmax(wei, dim=-1)
# xbow2 = wei @ x
# print(xbow2)

out = wei @ x
