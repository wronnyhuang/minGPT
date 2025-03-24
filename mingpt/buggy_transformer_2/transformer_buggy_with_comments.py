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
        # BUG: Using mean and var across all dimensions, not just the last one
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=True)  # BUG: Using unbiased=True instead of False
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # BUG: Missing the square root in the scale factor
        self.scale = 1.0 / self.head_dim
        
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
        # BUG: Incorrect reshaping - should be [..., self.num_heads, self.head_dim]
        q = self.q_proj(x).reshape(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply causal mask
        if mask is None:
            # BUG: Upper triangular mask is created with diagonal=0 which removes the current token attention
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=0).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # BUG: Incorrect masking - using True to mask values which should be False
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        out = torch.matmul(attn_weights, v)  # [B, num_heads, seq_len, head_dim]
        # BUG: Incorrect transpose order - should be 1, 2
        out = out.transpose(2, 1).reshape(batch_size, seq_len, dim)
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
        # BUG: Using ReLU instead of GELU activation
        x = F.relu(self.fc1(x))
        # BUG: Dropout applied to input of fc2 rather than output
        x = self.fc2(self.dropout(x))
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
        # BUG: Pre-norm transformer block applying norm after attention, should be before
        x = x + self.attn(x, mask)
        x = self.attn_norm(x)
        
        # BUG: Similar issue with FFN, should apply norm before FFN
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x

class ShardedTransformer(nn.Module):
    """
    Transformer model implementation with model sharding capability
    """
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        # BUG: vocab_size and block_size are not stored as class attributes
        
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
        # BUG: Weight tying missing - token_embedding.weight should be shared with lm_head
        self.lm_head = nn.Linear(dim, vocab_size)
        
        # Initialize parameters
        # BUG: Not applying initialization to the model
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # BUG: Using too large a standard deviation for initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
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
        
        # BUG: Missing sequence length check vs block_size
        
        # Get token embeddings
        token_emb = self.token_embedding(idx)  # [batch_size, seq_len, dim]
        
        # Get position embeddings
        # BUG: Not starting positions from 0
        positions = torch.arange(1, seq_len+1, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # [1, seq_len, dim]
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # BUG: Creating incorrect causal mask - should use upper triangular with diagonal=1
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
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
            # BUG: Not setting ignore_index for cross_entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
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
        # BUG: Missing @torch.no_grad() decorator
        
        # BUG: Using undefined self.vocab_size attribute
        vocab_size = self.lm_head.weight.size(0)
        
        for _ in range(max_new_tokens):
            # BUG: Not properly handling sequences longer than block_size
            
            # Get predictions
            logits, _ = self(idx)
            
            # Focus on the last token's predictions
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # BUG: Taking the most likely token instead of sampling
            _, next_token = torch.max(probs, dim=-1, keepdim=True)
            
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
    
    # BUG: Not moving all of the required modules to the first device
    model.token_embedding.to(devices[0])
    model.position_embedding.to(devices[0])
    # Missing model.dropout
    
    # Distribute transformer layers across devices
    num_layers = len(model.layers)
    # BUG: Incorrect calculation of layers_per_device
    layers_per_device = num_layers // len(devices)
    
    for i, layer in enumerate(model.layers):
        # BUG: Incorrect device assignment formula
        device_idx = i % len(devices)
        layer.to(devices[device_idx])
    
    # BUG: Not moving the final modules to the last device
    model.norm.to(devices[0])
    model.lm_head.to(devices[0])
    
    return model 