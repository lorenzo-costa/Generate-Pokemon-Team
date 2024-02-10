# imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import json

# ------------
# model definition

#multi head self attention module
class SelfAttention(nn.Module): 
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.attn = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.n_embd = n_embd
        self.n_head = n_head
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
    
    def forward(self, x):
        B,T,C = x.size()
        
        #calculating query, key, values
        q, k, v = self.attn(x).split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.proj(y))
        return y

#multi layer perceptron modele, with ReLU activation and dropout layer
class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# One block of the model 8¡(selfattention, MLP, 2x Layer Norm)
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.sa = SelfAttention(n_embd, n_head, dropout, block_size)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x))
        return x

# GPT implementation (2x embedding layers, 4 blocks, layernorm, dropour, final linear head)
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table  = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size = block_size
        
        self.apply(self._init_weights)
    
    # initializing models weight in a more clever manner to jumpstart training
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emd = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = self.drop(tok_emd + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        if targets is None:
            loss = None
            logits = self.lm_head(x[:, [-1], :])
        else:
            logits = self.lm_head(x) # (B, T, vocab_size)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature = 1): #lower temp more "diversity" aaa
        block_size = self.block_size
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]/temperature # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx   

def train(model, train, val, optimizer, max_iters, batch_size, block_size, eval_iters, eval_interval = 400):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = model.to(device)
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == (max_iters-1):
            losses = estimate_loss(m, train, val, batch_size, block_size, eval_iters)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train, block_size, batch_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return m

# ------------
# data loading
def get_batch(data, block_size, batch_size): #removed split
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# utility function to estimate intermediate loss
@torch.no_grad() 
def estimate_loss(model, train, val, batch_size, block_size, eval_iters): 
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            d = train
        else:
            d = val
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(d, block_size, batch_size) 
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
if __name__ == '__main__':
    # hyperparameters
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    max_iters = 15000 # number of iterations
    eval_interval = max_iters/30 # how many times the thing is printed
    learning_rate = 1.6e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 100
    n_embd = 48
    n_head = 4
    n_layer = 8
    dropout = 0.2
    torch.manual_seed(42)

    # ------------
    #data loader
    with open('names.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text+" -.'")))
    vocab_size = len(chars)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    with open('encoder', 'w') as f:
        json.dump(stoi, f)

    model = GPT(
    vocab_size=vocab_size,
    block_size=block_size, # what is the maximum context length for predictions?
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    m = train(model, train_data, val_data, optimizer, max_iters, batch_size,
    block_size, eval_iters)

    print('begin fine-tuning\n')

    with open('pokemon_names.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    data_pkm = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.8*len(data_pkm))
    train_data_pkm = data_pkm[:n]
    val_data_pkm = data_pkm[n:]

    max_iters = 800
    learning_rate = 1e-4
    dropout = 0.4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    m = train(model, train_data_pkm, val_data_pkm, optimizer, max_iters, batch_size, block_size, eval_iters, eval_interval)

    # generate from the model
    context = torch.tensor(encode("\n"), dtype=torch.long, device=device).view(1,-1)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

    torch.save(model.state_dict(), 'model_weights_pkm.pth')
# ~2.5M parameters