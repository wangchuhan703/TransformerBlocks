# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd=64, n_head=2):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert self.head_dim * n_head == n_embd, "Embedding size must be divisible by number of heads"

        # Define the linear layers for Q, K, V transformations
        self.q_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.k_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.v_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.out_linear = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, mask=None, mask_p=None):   # x.shape == (batch_size, block_size, n_embd)
        batch_size, seq_len, n_embd = x.size()

        Q = self.q_linear(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        # Q.shape == K.shape == V.shape == (batch_size, n_head, seq_len, head_dim)

        # Calculate attention scores (QK^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores.shape == (batch_size, n_head, seq_len, seq_len)

        if mask_p is not None:
            scores = scores + mask_p
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)

        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        # scores.shape == (batch_size, n_head, seq_len, head_dim) --> (batch_size, seq_len, n_head, head_dim)
        #                                                         --> (batch_size, seq_len, n_embd)

        # Final linear transformation
        output = self.out_linear(attn_output)
        # output.shape == (batch_size, block_size, n_embd) == x.shape

        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(n_embd, n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expansion factor for FFN
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask_p=None):   # x.shape == (batch_size, block_size, n_embd)
        x_norm = self.layernorm1(x)
        attn_output, attn_weights = self.attention(x_norm, mask_p = mask_p)
        x = x + attn_output  # Residual connection

        # Pre-LayerNorm before feed-forward network
        x_norm = self.layernorm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output  # Residual connection
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, block_size, vocab_size=10000):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([TransformerEncoderLayer(n_embd, n_head) for _ in range(n_layer)])

        self.alibi_slopes = self._get_alibi_slopes(n_head)
        self.n_head = n_head
        self.block_size = block_size

    def _get_alibi_slopes(self, n_head):
        """
        Generate slopes for ALiBi as per the paper, where each head has a different bias slope.
        The slopes follow a geometric sequence, e.g., [1.0, 0.5, 0.25, ..., 1/2^(n_head-1)].
        """
        return torch.tensor([1 / (2 ** i) for i in range(n_head)])


    def forward(self, x, mask=None):
        # Embedding
        token_embeddings = self.token_embedding(x)  # x.shape == (batch_size, block_size, n_embd)

        # chuhan:
        # 1. default positional encoding
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        position_embeddings = self.position_embedding(positions)  # shape == (1, block_size, n_embd)
        x = token_embeddings + position_embeddings  # x.shape == (batch_size, block_size, n_embd)

        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x)    # x.shape == (batch_size, block_size, n_embd)
            attn_maps.append(attn_weights)
        # end of 1. default positional encoding

        # # 3. ALiBi positional encoding
        # batch_size, seq_len = x.size()
        # alibi_biases = torch.zeros((self.n_head, seq_len, seq_len), device=x.device)
        # for h in range(self.n_head):
        #     slope = self.alibi_slopes[h]
        #     alibi_biases[h] = torch.arange(seq_len, device=x.device).view(1, -1) - torch.arange(seq_len, device=x.device).view(-1, 1)
        #     alibi_biases[h] = alibi_biases[h] * slope
        #     alibi_biases[h].tril_()  # Only apply bias to lower triangle to respect causality
        #
        # alibi_biases = alibi_biases.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, n_head, seq_len, seq_len)
        #
        # x = token_embeddings  # Initialize x with token embeddings
        # attn_maps = []
        # for layer in self.layers:
        #     x, attn_weights = layer(x, mask_p=alibi_biases)  # Passing ALiBi as mask
        #     attn_maps.append(attn_weights)
        # # end of 3. ALiBi positional encoding

        # Mean pooling across sequence dimension
        x = x.mean(dim=1)   # x.shape == (batch_size, n_embd)
        return x, attn_maps




class FeedForwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):   # x.shape == (batch_size=16, n_embd=64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(n_embd, n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None, mask_p=None):  # x.shape == (batch_size, block_size, n_embd)
        # Masked self-attention layer
        attn_output, attn_weights = self.attention(x, mask=mask, mask_p=None)   # x.shape == (batch_size, block_size, n_embd)
        x = self.layernorm1(x + attn_output)

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + ff_output)
        return x, attn_weights     # x.shape == (batch_size, block_size, n_embd)


class TransformerDecoder(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, block_size, vocab_size):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([TransformerDecoderLayer(n_embd, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.alibi_slopes = self._get_alibi_slopes(n_head)
        self.n_head = n_head
        self.block_size = block_size

    def _get_alibi_slopes(self, n_head):
        """
        Generate slopes for ALiBi as per the paper, where each head has a different bias slope.
        The slopes follow a geometric sequence, e.g., [1.0, 0.5, 0.25, ..., 1/2^(n_head-1)].
        """
        return torch.tensor([1 / (2 ** i) for i in range(n_head)])

    def forward(self, x, targets=None, mask=None):  # x.shape == (batch_size, block_size) == targets.shape
        token_embeddings = self.token_embedding(x)
        seq_len = x.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0)

        # chuhan
        # 1. default positional encoding
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings   # x.shape == (batch_size, block_size, n_embd)

        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)    # x.shape == (batch_size, block_size, n_embd)
            attn_maps.append(attn_weights)
        # end of 1. default positional encoding

        # # 3. ALiBi positional encoding
        # batch_size, seq_len = x.size()
        # alibi_biases = torch.zeros((self.n_head, seq_len, seq_len), device=x.device)
        # for h in range(self.n_head):
        #     slope = self.alibi_slopes[h]
        #     alibi_biases[h] = torch.arange(seq_len, device=x.device).view(1, -1) - torch.arange(seq_len, device=x.device).view(-1, 1)
        #     alibi_biases[h] = alibi_biases[h] * slope
        #     alibi_biases[h].tril_()  # Only apply bias to lower triangle to respect causality
        #
        # alibi_biases = alibi_biases.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, n_head, seq_len, seq_len)
        #
        # x = token_embeddings  # Initialize x with token embeddings
        # attn_maps = []
        # for layer in self.layers:
        #     x, attn_weights = layer(x, mask=mask, mask_p=alibi_biases)  # Passing ALiBi as mask
        #     attn_maps.append(attn_weights)
        # # end of 3. ALiBi positional encoding


        logits = self.lm_head(x)   # x.shape == (batch_size, block_size, vocab_size)
        if targets is None:
            return logits, attn_maps
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            return loss
