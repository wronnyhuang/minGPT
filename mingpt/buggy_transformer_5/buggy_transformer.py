import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ShardedTensor:
    def __init__(self, tensor, num_shards=2):
        self.num_shards = num_shards
        self.shards = torch.chunk(tensor, num_shards, dim=-1)
        self.requires_grad = False  
        self.grad = None
    
    def __getitem__(self, idx):
        return self.shards[idx]
    
    def cat(self):
        return torch.cat(self.shards, dim=-1)
    
    def backward(self, grad):
        """Custom backward function for sharded tensors"""
        if self.requires_grad:
            # Split the gradient into shards
            shard_grads = torch.chunk(grad, self.num_shards, dim=-1)
            
            # Store gradients for each shard
            self.grad = torch.cat([sg.detach() for sg in shard_grads], dim=-1)
            
            # Return the combined gradient for further backprop
            return self.grad
        return None

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Project keys, queries, and values for all heads
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # Dropout for attention
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        
        # Project and reshape
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        
        # Create a sharded version of the value tensor
        v_sharded = ShardedTensor(v.contiguous())
        
        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out_shards = []
        for i in range(v_sharded.num_shards):
            shard = v_sharded[i]
            out_shard = torch.matmul(attn, shard)
            out_shards.append(out_shard)
        
        out = torch.cat(out_shards, dim=0)
        
        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.resid_dropout(self.proj(out))
        
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim, ffn_dim)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = self.ln1(x)
        x = x + self.ffn(x)
        x = self.ln2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        # Sinusoidal positional embedding
        self.positional_embedding = self.create_sinusoidal_embeddings(block_size, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim) 
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(dim)
        
        # Output projection
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
            if module.bias is not None:
                torch.nn.init.ones_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def create_sinusoidal_embeddings(self, block_size, dim):
        """Create sinusoidal positional embeddings"""
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(1000.0) / dim))
        
        pe = torch.zeros(block_size, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, idx, targets=None):
        """
        idx: [batch_size, seq_len]
        targets: [batch_size, seq_len]
        returns
          logits: [batch_size, seq_len, vocab_size]
          loss: scalar
        """
        batch_size, seq_len = idx.size()
        assert seq_len <= self.block_size, f"Input sequence length ({seq_len}) exceeds model's context length ({self.block_size})"
        
        # Get token embeddings
        token_embeddings = self.token_embedding(idx)  # [batch_size, seq_len, dim]
        
        # Add positional embeddings
        positional_embeddings = self.positional_embedding[:seq_len, :]  # [seq_len, dim]
        x = token_embeddings + positional_embeddings  # [batch_size, seq_len, dim]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), reduction='mean')
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Runs inference.
        idx: [batch_size, seq_len]
        max_new_tokens: integer
        returns
          idx: [batch_size, seq_len+max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Truncate if context is too long
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self.forward(idx_cond, None)
            
            # Get the logits for the last token
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # Append sampled token to the sequence
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx 