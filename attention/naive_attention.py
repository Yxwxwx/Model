import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.sqrt_d_model = d_model**-0.5

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        """
        MaskedAttention(Q, K, V, M)=softmax(QK^T / sqrt(dk) + M) * V
        """
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.einsum("id, jd -> ij", q, k) * self.sqrt_d_model
        if mask is not None:
            sims = sims.masked_fill(mask=mask, value=float("-inf"))

        attention_percents = F.softmax(sims, dim=-1)
        out = torch.einsum("ij, jd -> id", attention_percents, v)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1, num_heads=2):
        super().__init__()

        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim) for _ in range(num_heads)]
        )
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        return torch.cat(
            [
                head(encodings_for_q, encodings_for_k, encodings_for_v, mask)
                for head in self.heads
            ],
            dim=self.col_dim,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encodings_for_q = torch.tensor([[1.16, 0.23], [0.57, 1.36], [4.41, -2.16]])
    encodings_for_k = torch.tensor([[1.16, 0.23], [0.57, 1.36], [4.41, -2.16]])
    encodings_for_v = torch.tensor([[1.16, 0.23], [0.57, 1.36], [4.41, -2.16]])

    torch.manual_seed(42)
    multi_head_attention = MultiHeadAttention(
        d_model=2, row_dim=0, col_dim=1, num_heads=2
    ).to(device)

    attention_scores = multi_head_attention(
        encodings_for_q.to(device),
        encodings_for_k.to(device),
        encodings_for_v.to(device),
    )
    print(attention_scores)
