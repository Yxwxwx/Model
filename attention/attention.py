import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelMultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 1. Three Linear Layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 2. Output Projection Layer
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # sqrt(d_k)
        self.register_buffer("scale", torch.tensor(self.d_k**-0.5))

    def forward(self, q_enc, k_enc, v_enc, mask=None):
        # batch_size (b) and sequence_length (l)
        b, l, _ = q_enc.size()

        # Full dimension Q, K, V (b, l, d_model)
        Q = self.W_q(q_enc)
        K = self.W_k(k_enc)
        V = self.W_v(v_enc)

        # Split d_model to (num_heads, d_k)
        # dims: (batch, seq_len, heads, head_dim) -> (b, l, h, d)
        Q = Q.view(b, l, self.num_heads, self.d_k)
        K = K.view(b, -1, self.num_heads, self.d_k)
        V = V.view(b, -1, self.num_heads, self.d_k)

        # ---------------------------------------------------------
        # Q: b(batch) i(seq_q) h(head) d(dim)
        # K: b(batch) j(seq_k) h(head) d(dim)
        #  contract d (dim) -> (b, h, i, j)
        # ---------------------------------------------------------
        sims = torch.einsum("bihd, bjhd -> bhij", Q, K) * self.scale

        if mask is not None:
            sims = sims.masked_fill(mask=mask, value=float("-inf"))

        attn_weights = F.softmax(sims, dim=-1)

        # ---------------------------------------------------------
        # W: (b, h, i, j)  V: (b, j, h, d)
        #  contract j (seq_k) -> (b, i, h, d)
        # ---------------------------------------------------------
        out = torch.einsum("bhij, bjhd -> bihd", attn_weights, V)
        out = out.reshape(b, l, self.d_model)

        final_out = self.W_o(out)

        return final_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # With Batch (Batch=1, Seq=3, Dim=4)
    encodings = torch.tensor(
        [[[1.16, 0.23, -0.5, 1.2], [0.57, 1.36, 0.8, -0.1], [4.41, -2.16, 0.3, 0.9]]],
        device=device,
    )

    torch.manual_seed(42)
    mha = ParallelMultiHeadAttention(d_model=4, num_heads=2).to(device)

    # Q, K, V from the same input, (Self-Attention)
    attention_out = mha(encodings, encodings, encodings)

    print(f"Output shape: {attention_out.shape}")
    print(attention_out)
