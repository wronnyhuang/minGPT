import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        # Create position encoding once and for all
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # BUG: Wrong divisor in the exponential term - should be using dim instead of dim/2
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / (dim/2)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        # BUG: Forgot to register as buffer - will be reset during model state_dict saving/loading
        self.pe = pe
        
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        seq_len = x.size(1)
        # BUG: Should be using .to(x.device) since pe might be on a different device
        return self.pe[:, :seq_len]

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, 3 * dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each [batch_size, num_heads, seq_len, head_dim]
        
        # Attention scores
        # BUG: Missing scaling factor in attention calculation
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            # BUG: Wrong masking value - should be -1e9 or float('-inf')
            scores = scores.masked_fill(mask == 0, -1)
        
        # Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        # BUG: Gradient accidentally disabled in attention weights
        attn_weights = attn_weights.detach()
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and project output
        # BUG: Wrong permutation order - breaks the reshaping logic
        attn_output = attn_output.permute(0, 2, 3, 1).contiguous()  # [batch_size, seq_len, head_dim, num_heads]
        attn_output = attn_output.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, dim]
        out = self.output_proj(attn_output)  # [batch_size, seq_len, dim]
        
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        # BUG: Using ReLU instead of GELU activation
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim, ffn_dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, dim]
        # BUG: Pre-norm architecture but used post-norm application
        x = x + self.attn(x, mask)
        x = self.ln1(x)
        # BUG: Pre-norm architecture but used post-norm application
        x = x + self.ffn(x)
        x = self.ln2(x)
        return x

class TransformerShardA(nn.Module):
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = SinusoidalPositionEmbedding(dim, block_size)
        
        # First half of the layers
        self.layers_first_half = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim) 
            for _ in range(num_layers // 2)
        ])
        
    def forward(self, idx):
        # idx: [batch_size, seq_len]
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        token_emb = self.token_embedding(idx)  # [batch_size, seq_len, dim]
        
        # Add positional encodings
        pos_emb = self.pos_emb(token_emb)
        # BUG: Positional embeddings should be added, not concatenated
        x = torch.cat([token_emb, pos_emb], dim=-1)[:, :, :token_emb.shape[-1]]
        
        # BUG: Causal mask is incorrectly created - should be in tril format
        mask = torch.ones(seq_len, seq_len, device=idx.device).view(1, 1, seq_len, seq_len)
        
        # Apply the first half of transformer blocks
        for layer in self.layers_first_half:
            x = layer(x, mask)
            
        return x

class TransformerShardB(nn.Module):
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size):
        super().__init__()
        # Second half of the layers
        self.layers_second_half = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim) 
            for _ in range(num_layers - num_layers // 2)
        ])
        
        self.ln_final = nn.LayerNorm(dim)
        # BUG: Weight sharing is missing - should tie weights with the embedding layer
        self.lm_head = nn.Linear(dim, vocab_size)
        
    def forward(self, x, mask=None):
        # Apply the second half of transformer blocks
        for layer in self.layers_second_half:
            x = layer(x, mask)
            
        # Final layer norm
        x = self.ln_final(x)
        
        # Language modeling head
        # BUG: Unintentionally detaching from computation graph before the final projection
        x = x.detach()
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        return logits

class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Sharded model
        self.shard_a = TransformerShardA(num_layers, num_heads, ffn_dim, dim, vocab_size, block_size)
        self.shard_b = TransformerShardB(num_layers, num_heads, ffn_dim, dim, vocab_size)
        
        # Initialize weights properly
        # BUG: Missing proper weight initialization
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # BUG: Wrong initialization variance - should be 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                # BUG: Should initialize biases to zero
                torch.nn.init.ones_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        idx: [batch_size, seq_len]
        targets: [batch_size, seq_len]
        returns
          logits: [batch_size, seq_len, vocab_size]
          loss: scalar
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.block_size, f"Input sequence length ({seq_len}) exceeds model's block size ({self.block_size})"
        
        # Create causal mask - but this is ignored in favor of the incorrect mask in shard_a
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).view(1, 1, seq_len, seq_len)
        
        # Pass through shard A
        x = self.shard_a(idx)
        
        # Pass through shard B
        logits = self.shard_b(x, mask)  # [batch_size, seq_len, vocab_size]
        
        # Compute loss if targets are provided
        if targets is not None:
            # BUG: Using mean reduction instead of sum for cross entropy
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), reduction='mean')
        else:
            loss = None
            
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        """runs inference.
        idx: [batch_size, seq_len]
        max_new_tokens: integer
        returns
          idx: [batch_size, seq_len+max_new_tokens]
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            # BUG: Incorrect handling of context - doesn't properly crop sequence
            idx_cond = idx
            logits, _ = self.forward(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # BUG: Using argmax instead of sampling - will produce deterministic outputs
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx 