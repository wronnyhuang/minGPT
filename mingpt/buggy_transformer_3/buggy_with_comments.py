import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # BUG: Disabling gradients by detaching the attention output prevents backpropagation
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.detach()  # BUG: This stops gradients from flowing through the attention layer
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size):
        super(TransformerModel, self).__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, dim)
        # BUG: Unintentionally disabling gradients for token embeddings will block learning for these parameters
        for param in self.token_embedding.parameters():
            param.requires_grad = False  # BUG: Gradients disabled for token embeddings
        self.pos_embedding = SinusoidalPositionalEmbedding(dim, max_len=block_size)

        # Manual sharding: splitting layers into two shards
        self.shard1 = nn.ModuleList()
        self.shard2 = nn.ModuleList()
        for i in range(num_layers):
            block = TransformerBlock(dim, num_heads, ffn_dim)
            if i % 2 == 0:
                self.shard1.append(block)
            else:
                self.shard2.append(block)

        self.ln_f = nn.LayerNorm(dim)
        # BUG: Off-by-one error: fc_out outputs vocab_size+1 instead of vocab_size, causing tensor shape mismatch
        self.fc_out = nn.Linear(dim, vocab_size + 1)  # BUG: Incorrect output dimension
        # BUG: Improper initialization: Zeroing weights prevents the model from learning useful representations
        self.fc_out.weight.data.zero_()  # BUG: Zero initialization

    def forward(self, idx, targets=None):
        # idx: [batch, seq_len]
        x = self.token_embedding(idx)  # [batch, seq_len, dim]
        x += self.pos_embedding(x)

        for block in self.shard1:
            x = block(x)
        for block in self.shard2:
            x = block(x)

        x = self.ln_f(x)
        logits = self.fc_out(x)  # [batch, seq_len, vocab_size+1]

        loss = None
        if targets is not None:
            # BUG: Detaching loss prevents gradients from flowing back, breaking backpropagation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)).detach()  # BUG: Detach loss
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Autoregressive generation
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # BUG: Concatenating along the wrong dimension (dim=0) will cause shape mismatch during generation
            idx = torch.cat((idx, next_token), dim=0)  # BUG: Wrong concatenation dimension
        return idx


if __name__ == "__main__":
    model = TransformerModel(num_layers=4, num_heads=2, ffn_dim=64, dim=32, vocab_size=1000, block_size=20)
    x = torch.randint(0, 1000, (2, 10))
    logits, loss = model(x, targets=x)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item() if loss is not None else None) 