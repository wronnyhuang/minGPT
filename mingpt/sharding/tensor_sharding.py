import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# Tensor Parallel Linear Layers
###############################################
# In this simple simulation, we split the weight matrix into two partitions.
# Each partition acts like a “shard” of the full weight. In real setups, these shards
# would reside on different GPUs.

class ColumnParallelLinear(nn.Module):
    """
    Splits the output dimension across partitions (columns).
    For an input of shape (..., in_features) and a weight of shape 
    (in_features, out_features), we split the out_features among partitions.
    """
    def __init__(self, in_features, out_features, num_partitions=2, bias=True):
        super(ColumnParallelLinear, self).__init__()
        assert out_features % num_partitions == 0, "Output features must be divisible by number of partitions"
        self.num_partitions = num_partitions
        self.out_features_per_partition = out_features // num_partitions
        
        # Create a parameter for each partition (each shard holds a slice of the full weight)
        self.weight_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, self.out_features_per_partition))
            for _ in range(num_partitions)
        ])
        if bias:
            self.bias_partitions = nn.ParameterList([
                nn.Parameter(torch.zeros(self.out_features_per_partition))
                for _ in range(num_partitions)
            ])
        else:
            self.bias_partitions = None

    def forward(self, x):
        # x: (..., in_features)
        # Compute the output for each partition and then concatenate along the last dimension.
        outputs = []
        for i in range(self.num_partitions):
            out_i = torch.matmul(x, self.weight_partitions[i])
            if self.bias_partitions is not None:
                out_i += self.bias_partitions[i]
            outputs.append(out_i)
        # Concatenate the partition outputs along the last dimension to form the full output.
        output = torch.cat(outputs, dim=-1)
        return output


class RowParallelLinear(nn.Module):
    """
    Splits the input dimension across partitions (rows).
    Here, the weight matrix of shape (in_features, out_features) is split along the in_features dimension.
    Each partition processes a slice of the input and then the partial results are summed.
    """
    def __init__(self, in_features, out_features, num_partitions=2, bias=True):
        super(RowParallelLinear, self).__init__()
        assert in_features % num_partitions == 0, "Input features must be divisible by number of partitions"
        self.num_partitions = num_partitions
        self.in_features_per_partition = in_features // num_partitions
        
        self.weight_partitions = nn.ParameterList([
            nn.Parameter(torch.randn(self.in_features_per_partition, out_features))
            for _ in range(num_partitions)
        ])
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        # x: (..., in_features) 
        # Split the input along the last dimension into num_partitions parts.
        xs = torch.split(x, self.in_features_per_partition, dim=-1)
        output = 0
        for i in range(self.num_partitions):
            output = output + torch.matmul(xs[i], self.weight_partitions[i])
        if self.bias is not None:
            output += self.bias
        return output

###############################################
# Transformer Components
###############################################
# We use the tensor parallel linear layers in the attention and feed-forward parts.
# In a Megatron-style transformer, the QKV projection and the output projection
# in the self-attention module are often split, and the feed-forward network as well.

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, tensor_parallel_size=2):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.tensor_parallel_size = tensor_parallel_size

        # Project Q, K, V all at once using a column parallel layer.
        # The output dimension is 3 * hidden_size (to be split into Q, K, V).
        self.qkv = ColumnParallelLinear(hidden_size, 3 * hidden_size, num_partitions=tensor_parallel_size)

        # The output projection is implemented using a row parallel layer.
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, num_partitions=tensor_parallel_size)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = x.size()
        # Compute QKV in one go. Shape: (batch, seq_len, 3 * hidden_size)
        qkv = self.qkv(x)
        # Split into Q, K, and V along the last dimension.
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        # Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to shape (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)
        # Concatenate heads to recover shape: (batch, seq_len, hidden_size)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.hidden_size)
        # Final projection
        output = self.out_proj(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4, tensor_parallel_size=2):
        super(FeedForward, self).__init__()
        self.inner_dim = hidden_size * mlp_ratio

        # First linear layer expands the dimension; we use a column parallel split.
        self.fc1 = ColumnParallelLinear(hidden_size, self.inner_dim, num_partitions=tensor_parallel_size)
        # Second linear layer projects back to hidden_size using a row parallel split.
        self.fc2 = RowParallelLinear(self.inner_dim, hidden_size, num_partitions=tensor_parallel_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, tensor_parallel_size=2):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, tensor_parallel_size=tensor_parallel_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, mlp_ratio=mlp_ratio, tensor_parallel_size=tensor_parallel_size)

    def forward(self, x):
        # Apply attention with a residual connection.
        x = x + self.attn(self.ln1(x))
        # Apply feed-forward network with a residual connection.
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, max_seq_len, tensor_parallel_size=2):
        super(SimpleTransformer, self).__init__()
        # Token embeddings and positional embeddings.
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Stack transformer blocks.
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, tensor_parallel_size=tensor_parallel_size)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        # Final output projection to vocabulary size.
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.size()
        token_embeds = self.token_embedding(x)
        # Create position indices and get corresponding embeddings.
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds

        # Pass through transformer layers.
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.output_layer(x)
        return logits

###############################################
# Testing the Model
###############################################
if __name__ == "__main__":
    # Set hyperparameters
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    hidden_size = 64
    num_heads = 8
    num_layers = 2
    max_seq_len = 512
    tensor_parallel_size = 2  # We are splitting weights into 2 partitions.

    # Create the model instance
    model = SimpleTransformer(vocab_size, hidden_size, num_heads, num_layers, max_seq_len, tensor_parallel_size)
    # Create a dummy input (random token IDs)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Forward pass
    logits = model(x)
    print("Logits shape:", logits.shape)  # Expected shape: (batch_size, seq_len, vocab_size)
