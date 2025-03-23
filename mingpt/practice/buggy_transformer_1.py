import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import time

# ============================================================================
# Toy Dataset: Pattern Recognition Task
# The task: given a sequence, determine if it contains a specific pattern
# ============================================================================

class PatternDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=20, vocab_size=10, pattern_length=3, split='train'):
        """
        Creates a dataset where the task is to identify if a specific pattern exists in the sequence.
        The label is 1 if the pattern exists, 0 otherwise.
        
        For training: random patterns are used
        For eval: new patterns not seen during training are used
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pattern_length = pattern_length
        self.split = split
        
        # Set different random seeds for train/eval to ensure different patterns
        seed = 42 if split == 'train' else 123
        np.random.seed(seed)
        
        # Generate patterns based on split
        if split == 'train':
            # Training patterns are in the first half of vocab space
            self.patterns = [np.random.randint(0, vocab_size//2, pattern_length) for _ in range(5)]
        else:
            # Eval patterns are in the second half of vocab space - tests generalization
            self.patterns = [np.random.randint(vocab_size//2, vocab_size, pattern_length) for _ in range(5)]
        
        # Generate sequences and labels
        self.sequences = []
        self.labels = []
        
        for _ in range(num_samples):
            # Generate a random sequence
            seq = np.random.randint(0, vocab_size, seq_length)
            
            # Decide whether to insert a pattern (50% chance)
            has_pattern = np.random.random() < 0.5
            label = 1 if has_pattern else 0
            
            if has_pattern:
                # Choose a random pattern and insert at a random position
                pattern = self.patterns[np.random.randint(0, len(self.patterns))]
                start_idx = np.random.randint(0, seq_length - pattern_length + 1)
                seq[start_idx:start_idx + pattern_length] = pattern
            
            self.sequences.append(seq)
            self.labels.append(label)
        
        # Convert to PyTorch tensors
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================================
# Transformer Model Implementation (with intentional bugs)
# ============================================================================

class BuggyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # Bug 1: Missing assertion that d_model is divisible by num_heads
        
        self.d_model = d_model
        self.num_heads = num_heads
        # Bug 2: Incorrect head dimension calculation (should be d_model // num_heads)
        self.head_dim = d_model / num_heads  # Should be integer division
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Bug 3: Dropout applied at wrong place
        self.dropout = nn.Dropout(dropout)
        
        # Bug 4: Wrong initialization
        # Should use appropriate gain, and Linear modules are already initialized
        for p in self.parameters():
            if p.dim() > 1:
                # This is redundant and applies wrong scaling
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshaping
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Bug 5: Incorrect reshaping for multi-head attention
        # Should convert to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, int(self.head_dim)).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, int(self.head_dim)).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, int(self.head_dim)).permute(0, 2, 1, 3)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        # Bug 6: Missing or incorrect scaling factor
        # Should be 1/sqrt(d_k)
        scaling = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))  # Wrong: should use head_dim
        scores = scores / scaling
        
        # Apply mask if provided
        if mask is not None:
            # Bug 7: Incorrect mask application
            # Mask should be properly broadcast and use -inf
            scores = scores.masked_fill(mask == 0, -1e4)  # Should use -inf
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        # Bug 8: Dropout applied before matrix multiplication
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        # Bug 9: Incorrect reshaping back
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_proj(output)
        
        return output


class BuggyPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Bug 10: Incorrect divisor in the exponential term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))  # Should be 10000.0
        
        # Bug 11: Swapped sine and cosine
        pe[:, 0::2] = torch.cos(position * div_term)  # Should be sine
        pe[:, 1::2] = torch.sin(position * div_term)  # Should be cosine
        
        pe = pe.unsqueeze(0)
        
        # Register buffer (correct)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Bug 12: Incorrect addition of positional encoding
        # Should add to input, not replace it
        return self.pe[:, :x.size(1)]  # Should be x + self.pe[:, :x.size(1)]


class BuggyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = BuggyMultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = F.relu
        
    def forward(self, src, src_mask=None):
        # Bug 13: Incorrect application of layer norm (should be applied before attention)
        # Correct: attn_output = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), src_mask)
        attn_output = self.self_attn(src, src, src, src_mask)
        
        # Bug 14: Incorrect residual connection (should add to input)
        # Correct: src = src + self.dropout(attn_output)
        src = self.dropout(attn_output)
        
        # Bug 15: Layer norm applied in wrong order
        src = self.norm1(src)
        
        # Feed-forward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # Bug 16: Missing dropout after second linear layer
        # Correct: ff_output = self.dropout(ff_output)
        
        # Bug 17: Second residual connection issue
        # Correct: src = src + ff_output
        src = ff_output
        
        src = self.norm2(src)
        return src


class BuggyTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BuggyTransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, src_mask=None):
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask)
            
        return output


class BuggyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, dim_feedforward, num_layers, num_classes, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Bug 18: Incorrect embedding scaling (should be sqrt(d_model))
        self.embed_scale = d_model  # Should be math.sqrt(d_model)
        
        # Positional encoding
        self.pos_encoding = BuggyPositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        self.transformer_encoder = BuggyTransformerEncoder(
            num_layers, d_model, num_heads, dim_feedforward, dropout)
        
        # Classification head
        # Bug 19: Missing dropout before classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, src_mask=None):
        # Get sequence length
        seq_len = x.size(1)
        
        # Embed tokens and positions
        # Bug 20: Incorrect scaling of embeddings
        x = self.embedding(x) * self.embed_scale  # Should be sqrt(d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_mask)
        
        # Bug 21: Incorrect pooling strategy for classification
        # Should average over sequence or use [CLS] token if implemented
        # Taking just the last token is problematic for this task
        x = x[:, -1]  # Should use mean or proper pooling
        
        # Apply classification head
        x = self.classifier(x)
        
        return x


# ============================================================================
# Training and Evaluation Scaffolding (bug free)
# ============================================================================

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Accumulate loss
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Return average loss and accuracy
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Accumulate loss
            total_loss += loss.item()
    
    # Return average loss and accuracy
    avg_loss = total_loss / len(eval_loader)
    accuracy = 100. * correct / total
    print(f'Evaluation: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy


def create_src_mask(src):
    # Create a mask for padding tokens (assumes 0 is pad token)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    return src_mask


# ============================================================================
# Main training loop
# ============================================================================

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    batch_size = 32
    vocab_size = 10
    d_model = 64
    num_heads = 4
    dim_feedforward = 256
    num_layers = 2
    num_classes = 2  # Binary classification
    dropout = 0.1
    
    # Learning parameters
    learning_rate = 0.001
    num_epochs = 10
    
    # Create datasets
    train_dataset = PatternDataset(num_samples=2000, seq_length=20, vocab_size=vocab_size, pattern_length=3, split='train')
    eval_dataset = PatternDataset(num_samples=500, seq_length=20, vocab_size=vocab_size, pattern_length=3, split='eval')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = BuggyTransformerClassifier(
        vocab_size, d_model, num_heads, dim_feedforward, num_layers, num_classes, dropout=dropout
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Before training evaluation
    print("Evaluation before training:")
    before_loss, before_acc = evaluate(model, eval_loader, criterion, device)
    
    # Training loop
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_accs = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
    
    # After training evaluation
    print("\nEvaluation after training:")
    after_loss, after_acc = evaluate(model, eval_loader, criterion, device)
    
    # # Plot results
    # epochs = range(1, num_epochs + 1)
    
    # plt.figure(figsize=(12, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    # plt.plot(epochs, eval_losses, 'r-', label='Evaluation Loss')
    # plt.title('Training and Evaluation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    # plt.plot(epochs, eval_accs, 'r-', label='Evaluation Accuracy')
    # plt.title('Training and Evaluation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.savefig('training_results.png')
    # plt.show()
    
    # print(f"\nBefore training - Evaluation accuracy: {before_acc:.2f}%")
    # print(f"After training  - Evaluation accuracy: {after_acc:.2f}%")
    
    # # This will only show generalization success if bugs are fixed!
    # if after_acc - before_acc > 20:
    #     print("\nModel is showing good generalization!")
    # else:
    #     print("\nModel is not generalizing well. There might be bugs in the implementation.")


if __name__ == "__main__":
    main() 