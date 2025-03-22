import torch
import math

# -------------------------------------------------------------------
# Custom Linear Module with manual forward/backward (no nn.Linear)
# -------------------------------------------------------------------
class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (batch, *, in_features)
        # weight: (out_features, in_features)
        # bias: (out_features,)
        ctx.save_for_backward(x, weight, bias)
        output = x.mm(weight.t()) + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(x)
        grad_bias = grad_output.sum(0)
        return grad_x, grad_weight, grad_bias

class CustomLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Simple weight initialization for educational purposes.
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2.0/in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return CustomLinearFunction.apply(x, self.weight, self.bias)

# -------------------------------------------------------------------
# Custom Softmax Function (applied on the last dimension)
# -------------------------------------------------------------------
class CustomSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # subtract max for numerical stability
        x_max, _ = x.max(dim=-1, keepdim=True)
        exp_x = (x - x_max).exp()
        sum_exp = exp_x.sum(dim=-1, keepdim=True)
        out = exp_x / sum_exp
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        # For each row: dX = out * (grad_output - sum(grad_output*out))
        sum_grad = (grad_output * out).sum(dim=-1, keepdim=True)
        grad_input = out * (grad_output - sum_grad)
        return grad_input

# -------------------------------------------------------------------
# Custom ReLU Function
# -------------------------------------------------------------------
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# -------------------------------------------------------------------
# Custom Layer Normalization Module
# (Assumes normalization is over the last dimension)
# -------------------------------------------------------------------
class CustomLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        # x shape: (..., features)
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False is used here for simplicity.
        std = (x.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
        normalized = (x - mean) / std
        out = gamma * normalized + beta
        ctx.save_for_backward(normalized, std, gamma)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        normalized, std, gamma = ctx.saved_tensors
        # dnormalized = grad_output * gamma
        dx_normalized = grad_output * gamma
        # Using the formula:
        # dx = (1/std) * (dx_normalized - mean(dx_normalized) - normalized * mean(dx_normalized * normalized))
        dx = (1.0 / std) * (dx_normalized - dx_normalized.mean(dim=-1, keepdim=True) 
                            - normalized * (dx_normalized * normalized).mean(dim=-1, keepdim=True))
        # Sum over all dimensions except the last one for gamma and beta gradients
        sum_dims = tuple(range(grad_output.dim()-1))
        dgamma = (grad_output * normalized).sum(dim=sum_dims)
        dbeta = grad_output.sum(dim=sum_dims)
        return dx, dgamma, dbeta, None

class CustomLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        normalized_shape: int or tuple; for transformer, typically the model dimension.
        """
        super().__init__()
        # Here we assume normalized_shape is an int (the last dimension size).
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return CustomLayerNormFunction.apply(x, self.gamma, self.beta, self.eps)

# -------------------------------------------------------------------
# Scaled Dot-Product Attention with manual backward propagation
# -------------------------------------------------------------------
class ScaledDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, d_k):
        # Q, K, V: (batch, heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn = CustomSoftmax.apply(scores)
        output = torch.matmul(attn, V)
        ctx.save_for_backward(Q, K, V, attn)
        ctx.d_k = d_k
        return output, attn

    @staticmethod
    def backward(ctx, grad_output, grad_attn_unused=None):
        Q, K, V, attn = ctx.saved_tensors
        d_k = ctx.d_k
        # Gradients for V: grad_V = attn^T @ grad_output
        grad_V = torch.matmul(attn.transpose(-2, -1), grad_output)
        # Gradient wrt attention weights from output: grad_attn = grad_output @ V^T
        grad_attn = torch.matmul(grad_output, V.transpose(-2, -1))
        # Backprop through softmax:
        # For each row, grad_scores = attn * (grad_attn - sum(grad_attn*attn))
        sum_grad = (grad_attn * attn).sum(dim=-1, keepdim=True)
        grad_scores = attn * (grad_attn - sum_grad)
        # Since scores were scaled by 1/sqrt(d_k)
        grad_scores = grad_scores / math.sqrt(d_k)
        # Gradients for Q and K from the scores
        grad_Q = torch.matmul(grad_scores, K)
        grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q)
        return grad_Q, grad_K, grad_V, None

# -------------------------------------------------------------------
# Multi-Head Attention Module
# -------------------------------------------------------------------
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model: the input (and output) feature dimension.
        num_heads: number of attention heads; d_model must be divisible by num_heads.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

        # Using our custom linear layers for query, key, value projections.
        self.W_q = CustomLinear(d_model, d_model)
        self.W_k = CustomLinear(d_model, d_model)
        self.W_v = CustomLinear(d_model, d_model)
        self.W_o = CustomLinear(d_model, d_model)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        # Split into heads and reshape:
        # -> (batch_size, seq_len, num_heads, d_k) then transpose to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Apply scaled dot-product attention (returns both output and attention weights)
        attn_output, attn_weights = ScaledDotProductAttention.apply(Q, K, V, self.d_k)
        # Merge heads: transpose back and reshape to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # Final linear projection
        output = self.W_o(attn_output)
        return output

# -------------------------------------------------------------------
# Position-wise Feed-Forward Network Module
# -------------------------------------------------------------------
class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        """
        d_model: model dimension.
        d_ff: hidden layer dimension in the feed-forward network.
        """
        super().__init__()
        self.linear1 = CustomLinear(d_model, d_ff)
        self.linear2 = CustomLinear(d_ff, d_model)

    def forward(self, x):
        # First linear layer -> ReLU -> second linear layer.
        return self.linear2(CustomReLU.apply(self.linear1(x)))

# -------------------------------------------------------------------
# Positional Encoding Module (fixed, not learnable)
# -------------------------------------------------------------------
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Creates fixed sinusoidal positional encodings.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        Returns x with positional encodings added.
        """
        return x + self.pe[:, :x.size(1)]

# -------------------------------------------------------------------
# Transformer Encoder Layer Module
# -------------------------------------------------------------------
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        A single Transformer encoder block.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = CustomLayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm2 = CustomLayerNorm(d_model)
        self.dropout = dropout  # For simplicity, dropout is not manually implemented here.

    def forward(self, x):
        # Self-attention sub-layer with residual connection and layer normalization.
        attn_output = self.self_attn(x)
        x = self.layer_norm1(x + attn_output)
        # Feed-forward sub-layer with residual connection and layer normalization.
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x

# @title Attention backprop
class AttentionFun(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, QKV, num_heads):
    b, t, d = x.shape
    qkv = x @ QKV.transpose(-1, -2)
    q, k, v = qkv.split(d, dim=-1)
    q = q.view(b, t, num_heads, d // num_heads).transpose(1, 2)
    k = k.view(b, t, num_heads, d // num_heads).transpose(1, 2)
    v = v.view(b, t, num_heads, d // num_heads).transpose(1, 2)
    logits = q @ k.transpose(-1, -2) / k.size(-1) ** 0.5
    scores = SoftMax.apply(logits)
    output = scores @ v
    output = output.transpose(1, 2).contiguous().view(b, t, d)
    ctx.save_for_backward(x, QKV, q, k, v, scores)
    return output

  def backward(ctx, grad_output):
    b, t, d = grad_output.shape
    x, QKV, q, k, v, scores = ctx.saved_tensors
    num_heads = k.size(1)

    grad_output = grad_output.view(b, t, num_heads, d // num_heads).transpose(1, 2)
    grad_scores = grad_output @ v.transpose(-1, -2)
    grad_v = scores.transpose(-1, -2) @ grad_output

    grad_logits = scores * (grad_scores - (scores * grad_scores).sum(dim=-1, keepdim=True))
    grad_q = grad_logits @ k / k.size(-1) ** 0.5
    grad_k = (q.transpose(-1, -2) @ grad_logits).transpose(-1, -2) / k.size(-1) ** 0.5

    grad_q = grad_q.transpose(1, 2).contiguous().view(b, t, d)
    grad_k = grad_k.transpose(1, 2).contiguous().view(b, t, d)
    grad_v = grad_v.transpose(1, 2).contiguous().view(b, t, d)
    grad_qkv = torch.cat([grad_q, grad_k, grad_v], dim=-1)
    grad_x = grad_qkv @ QKV
    grad_QKV = (x.transpose(-1, -2) @ grad_qkv).transpose(-1, -2)
    return grad_x, grad_QKV, None



# -------------------------------
# Example usage (for testing)
# -------------------------------
if __name__ == '__main__':
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    d_ff = 32

    # Sample input
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    # Positional encoding
    pos_enc = PositionalEncoding(d_model)
    x = pos_enc(x)

    # Transformer encoder layer
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    output = encoder_layer(x)
    
    # Compute a dummy loss and perform backward propagation
    loss = output.sum()
    loss.backward()
    
    print("Forward and backward passes completed successfully.")


