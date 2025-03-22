# @title Put it all together
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.practice.my_functions import *

def printt(*tensors):
  return [f'{tensor.dtype}, {tensor.shape}' for tensor in tensors]

class Attention(nn.Module):
  def __init__(self, num_heads, dim):
    super().__init__()
    self.num_heads = num_heads
    self.QKV = nn.Linear(dim, 3 * dim)
  def forward(self, x):
    b, t, d = x.shape
    qkv = self.QKV(x)
    q, k, v = torch.split(qkv, d, dim=-1)
    q = q.view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)
    k = k.view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)
    v = v.view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)
    logits = q @ k.transpose(-1, -2) / d ** 0.5
    logits = torch.where(torch.tril(torch.ones_like(logits)) == 1, logits, -float('inf')) 
    attn = F.softmax(logits, dim=-1)
    x = attn @ v
    x = x.transpose(1, 2).contiguous().view(b, t, d)
    return x

class Ffn(nn.Module):
  def __init__(self, ffn_dim, dim):
    super().__init__()
    self.fc1 = nn.Linear(dim, ffn_dim)
    self.fc2 = nn.Linear(ffn_dim, dim)
  def forward(self, x):
    x = self.fc1(x)
    x = F.gelu(x)
    x = self.fc2(x)
    x = F.gelu(x)
    return x

class LayerNorm(nn.Module):
  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.bias = torch.nn.Parameter(torch.zeros(dim))
  def forward(self, x):
    return LayerNormFun.apply(x, self.weight, self.bias, self.eps)
    

class Block(nn.Module):
  def __init__(self, num_heads, ffn_dim, dim):
    super().__init__()
    self.attn = Attention(num_heads, dim)
    self.ffn = Ffn(ffn_dim, dim)
    self.ln1 = nn.LayerNorm(dim)
    self.ln2 = nn.LayerNorm(dim)
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x


class Transformer(nn.Module):
  def __init__(self, num_layers, num_heads, ffn_dim, dim, vocab_size, block_size):
    super().__init__()
    self.block_size = block_size
    self.emb = nn.Embedding(vocab_size, dim)
    self.pos_emb = nn.Embedding(block_size, dim)
    self.layers = nn.ModuleList([Block(num_heads, ffn_dim, dim) for _ in range(num_layers)])
    self.lnf = nn.LayerNorm(dim)
    self.lm_head = nn.Linear(dim, vocab_size)

  def forward(self, idx, targets=None):
    b, t = idx.shape
    x = self.emb(idx)
    pos = self.pos_emb(torch.arange(t))
    x = x + pos
    for layer in self.layers:
      x = layer(x)
    x = self.lnf(x)
    logits = self.lm_head(x)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return logits, loss

  def configure_optimizers(self, train_config):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
    return optimizer

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
      """
      Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
      the sequence max_new_tokens times, feeding the predictions back into the model each time.
      Most likely you'll want to make sure to be in model.eval() mode of operation for this.
      """
      for _ in range(max_new_tokens):
          # if the sequence context is growing too long we must crop it at block_size
          idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
          # forward the model to get the logits for the index in the sequence
          logits, _ = self(idx_cond)
          # pluck the logits at the final step and scale by desired temperature
          logits = logits[:, -1, :] / temperature
          # optionally crop the logits to only the top k options
          if top_k is not None:
              v, _ = torch.topk(logits, top_k)
              logits[logits < v[:, [-1]]] = -float('Inf')
          # apply softmax to convert logits to (normalized) probabilities
          probs = F.softmax(logits, dim=-1)
          # either sample from the distribution or take the most likely element
          if do_sample:
              idx_next = torch.multinomial(probs, num_samples=1)
          else:
              _, idx_next = torch.topk(probs, k=1, dim=-1)
          # append sampled index to the running sequence and continue
          idx = torch.cat((idx, idx_next), dim=-1)
      return idx
  

