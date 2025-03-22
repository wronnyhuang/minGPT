# @title Import
import torch
import torch.nn as nn
from torch.nn import functional as F

def printt(*tensors):
  return [f'{tensor.dtype}, {tensor.shape}' for tensor in tensors]


# @title Layer norm
class LayerNormFun(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = (x.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()
    normalized = (x - mean) / std
    output = weight * normalized + bias
    ctx.save_for_backward(normalized, std, weight)
    return output
  @staticmethod
  def backward(ctx, grad_output):
    normalized, std, weight = ctx.saved_tensors
    grad_normalized = grad_output * weight
    grad_input = 1 / std * (grad_normalized - grad_normalized.mean(dim=-1, keepdim=True) - normalized * (grad_normalized * normalized).mean(dim=-1, keepdim=True))
    sum_dims = tuple(range(grad_output.dim() - 1))
    grad_weight = (grad_output * normalized).sum(dim=sum_dims)
    grad_bias = grad_output.sum(dim=sum_dims)
    return grad_input, grad_weight, grad_bias, None


# @title Linear layer
class Linear(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias):
    output = x @ weight.transpose(0, 1) + bias
    ctx.save_for_backward(x, weight, bias)
    return output
  @staticmethod
  def backward(ctx, grad_output):
    x, weight, bias = ctx.saved_tensors
    grad_input = grad_output @ weight
    grad_weight = (x.transpose(0, 1) @ grad_output).transpose(0, 1)
    sum_dim = tuple(range(grad_output.dim() - 1))
    grad_bias = grad_output.sum(dim=sum_dim)
    return grad_input, grad_weight, grad_bias
  

# @title Softmax
class SoftMax(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    x_max, _ = x.max(dim=-1, keepdim=True)
    x = x - x_max
    sum_exp = torch.logsumexp(x, dim=-1, keepdim=True).exp()
    output = x.exp() / sum_exp
    ctx.save_for_backward(output)
    return output
  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    grad_input = output * (grad_output - (output * grad_output).sum(dim=-1, keepdim=True))
    return grad_input

# @title Cross entropy
class CrossEntropy(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, one_hot_labels):
    probs = SoftMax.apply(x)
    loss = -(one_hot_labels * torch.log(probs)).sum(dim=-1).mean()
    ctx.save_for_backward(probs, one_hot_labels)
    return loss
  @staticmethod
  def backward(ctx, grad_output):
    probs, one_hot_labels = ctx.saved_tensors
    grad_input = grad_output * (probs - one_hot_labels)
    return grad_input, None

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
    logits = torch.where(torch.tril(torch.ones_like(logits)) == 1, logits, -float('inf'))
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
