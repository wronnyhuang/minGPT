import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBackward(torch.autograd.Function):
    """Custom autograd function for attention calculation with manual backward pass"""
    
    @staticmethod
    def forward(ctx, q, k, v, mask=None, dropout_p=0.0):
        """
        q, k, v: [..., seq_len, head_dim]
        mask: [..., seq_len, seq_len] or None
        Returns attention outputs
        """
        # Scaled dot-product attention
        d_k = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout if probability > 0
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        
        output = torch.matmul(attn_weights, v)
        
        # Save for backward
        ctx.save_for_backward(q, k, v, attn_weights)
        ctx.mask = mask
        ctx.dropout_p = dropout_p
        
        return output, attn_weights

    @staticmethod
    def backward(ctx, grad_output, grad_weights):
        """Manual backward pass for attention calculation"""
        q, k, v, attn_weights = ctx.saved_tensors
        mask = ctx.mask
        dropout_p = ctx.dropout_p
        
        d_k = q.size(-1)
        
        # Gradient w.r.t. output of attention weights
        grad_attn_weights = torch.matmul(grad_output, v.transpose(-2, -1))
        
        # Account for softmax in backward pass
        grad_attn_scores = attn_weights * (grad_attn_weights - (attn_weights * grad_attn_weights).sum(dim=-1, keepdim=True))
        
        # If we used mask in forward, account for it
        if mask is not None:
            grad_attn_scores = grad_attn_scores.masked_fill(mask == 0, 0)
        
        # Gradient w.r.t. q, k, v
        grad_q = torch.matmul(grad_attn_scores, k) / math.sqrt(d_k)
        grad_k = torch.matmul(grad_attn_scores.transpose(-2, -1), q) / math.sqrt(d_k)
        grad_v = torch.matmul(attn_weights.transpose(-2, -1), grad_output)
        
        return grad_q, grad_k, grad_v, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = dropout
        self.attention = AttentionBackward.apply
    
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply our custom attention with manual backprop
        attn_output, _ = self.attention(q, k, v, mask, self.dropout)
        
        # Reshape and project back to original dimension
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, dim)
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, ffn_dim, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Token embedding + positional embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(block_size, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        """
        idx: [batch_size, seq_len]
        targets: [batch_size, seq_len]
        returns
          logits: [batch_size, seq_len, vocab_size]
          loss: scalar    
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.block_size, f"Input sequence length {seq_len} exceeds block size {self.block_size}"
        
        # Create position indices and mask
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Get token embeddings and add positional embeddings
        token_emb = self.token_embedding(idx)  # [batch_size, seq_len, dim]
        pos_emb = self.position_embedding(positions)  # [1, seq_len, dim]
        x = token_emb + pos_emb
        
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).view(1, 1, seq_len, seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Apply final layer norm and project to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)  # [batch_size, seq_len, vocab_size]
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """runs inference.
        idx: [batch_size, seq_len]
        max_new_tokens: integer
        returns
          idx: [batch_size, seq_len+max_new_tokens]
        """
        # idx starts as (batch_size, seq_len)
        for _ in range(max_new_tokens):
            # Truncate sequence if it's getting too long
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on the last token
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get next token (greedy sampling)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch_size, 1)
            
            # Append to the sequence
            idx = torch.cat([idx, next_token], dim=1)  # (batch_size, seq_len+1)
        
        return idx 