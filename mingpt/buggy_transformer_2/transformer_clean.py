import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """Layer normalization module"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.size()
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        out = torch.matmul(attn_weights, v)  # [B, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.resid_dropout(self.out_proj(out))
        
        return out

class FeedForward(nn.Module):
    """Feed forward network"""
    def __init__(self, dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x

class TransformerBlock(nn.Module):
    """Transformer block with sharding support"""
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn_norm = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
        self.ffn_norm = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x = x + self.attn(self.attn_norm(x), mask)
        # FFN with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ShardedTransformer(nn.Module):
    """
    Transformer model implementation with model sharding capability
    """
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        # Position embedding
        self.position_embedding = nn.Embedding(block_size, dim)
        
        # Setup for sharding the model across layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.lm_head = nn.Linear(dim, vocab_size)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx, targets=None):
        """
        idx: [batch_size, seq_len]
        targets: [batch_size, seq_len]
        returns
          logits: [batch_size, seq_len, vocab_size]
          loss: scalar
        """
        device = idx.device
        batch_size, seq_len = idx.size()
        
        # Check sequence length
        assert seq_len <= self.block_size, f"Input sequence length ({seq_len}) exceeds block size ({self.block_size})"
        
        # Get token embeddings
        token_emb = self.token_embedding(idx)  # [batch_size, seq_len, dim]
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # [1, seq_len, dim]
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross entropy
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Runs inference.
        idx: [batch_size, seq_len]
        max_new_tokens: integer
        returns
          idx: [batch_size, seq_len+max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop sequence if it becomes longer than block_size
            if idx.size(1) > self.block_size:
                idx = idx[:, -self.block_size:]
                
            # Get predictions
            logits, _ = self(idx)
            
            # Focus on the last token's predictions
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # Append sampled token to the sequence
            idx = torch.cat([idx, next_token], dim=1)
            
        return idx

# Helper function to shard a model across multiple devices
def shard_model(model, devices):
    """
    Distribute model layers across multiple devices
    """
    if len(devices) == 1:
        model.to(devices[0])
        return model
    
    # Assign embedding and first norm to the first device
    model.token_embedding.to(devices[0])
    model.position_embedding.to(devices[0])
    model.dropout.to(devices[0])
    
    # Distribute transformer layers across devices
    num_layers = len(model.layers)
    layers_per_device = max(1, num_layers // len(devices))
    
    for i, layer in enumerate(model.layers):
        device_idx = min(i // layers_per_device, len(devices) - 1)
        layer.to(devices[device_idx])
    
    # Assign final norm and lm_head to the last device
    model.norm.to(devices[-1])
    model.lm_head.to(devices[-1])
    
    return model 