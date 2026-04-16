import jax
import jax.numpy as jnp
import flax.nnx as nnx
import grain.python as grain

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


def load_stories_from_file(file_path: Path, max_stories: int = 1000) -> List[str]:
    """
    Streams the file and splits stories, ensuring each story ends with the <|endoftext|> separator.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stories = []
    separator = "<|endoftext|>"
    buffer = ""

    print(f"Loading data from {file_path} (max {max_stories} stories)")

    with file_path.open("r", encoding="utf-8") as f:
        while len(stories) < max_stories:
            chunk = f.read(1024 * 1024)  # 1MB chunk
            if not chunk:
                # Handle the final remaining content in the buffer
                remaining = buffer.strip()
                if remaining:
                    # Ensure the last story also has the EOS token
                    if not remaining.endswith(separator):
                        remaining += separator
                    stories.append(remaining)
                break

            buffer += chunk

            # Extract stories and re-attach the separator
            while separator in buffer and len(stories) < max_stories:
                story_content, buffer = buffer.split(separator, 1)
                # Combine the content with the separator for the tokenizer
                stories.append(story_content.strip() + separator)

    print(f"Loaded {len(stories)} stories")
    return stories


# Create the embedding layers
class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


# Build the Transformer block
class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x, mask=None):
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        return x


# Define the model configuration
class MiniGPT(nnx.Module):
    def __init__(
        self,
        maxlen,
        vocab_size,
        embed_dim,
        num_heads,
        feed_forward_dim,
        num_transformer_blocks,
        *,
        rngs,
    ):
        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs
        )
        self.transformer_blocks = nnx.List(
            [
                TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
                for _ in range(num_transformer_blocks)
            ]
        )
        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)
        return logits


class StoryDataset:
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={"<|endoftext|>"})

        if len(tokens) > self.maxlen:
            tokens = tokens[: self.maxlen]

        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens


def print_sampler_example(sampler, name):
    print(f"\n{name} sampler example:")
    for i, idx in enumerate(sampler):
        print(f"Record {i}: {idx}")


# Build a data loader
def create_dataloader(
    stories,
    tokenizer,
    maxlen,
    batch_size,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0,
):
    dataset = StoryDataset(stories, maxlen, tokenizer)
    estimated_batches = len(dataset) // batch_size

    # Build a data iterator
    sampler = grain.IndexSampler(
        num_records=len(dataset),  # 1000 stories
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        # Batch sequence into fixed-size arrays
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=worker_count,
    )

    print(f"Estimated batches per epoch: {estimated_batches}")
    print(f"Created DataLoader with batch_size={batch_size}, maxlen={maxlen}")

    return dataloader, estimated_batches


def load_and_preprocess_data(
    file_path, tokenizer, batch_size, maxlen, max_stories, shuffle, seed
):
    stories = load_stories_from_file(file_path, max_stories=max_stories)
    dataloader, estimated_batches = create_dataloader(
        stories, tokenizer, maxlen, batch_size, shuffle, seed, worker_count=0
    )

    return dataloader, estimated_batches


if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    model = MiniGPT(
        maxlen=128,
        vocab_size=tokenizer.n_vocab,
        embed_dim=192,
        num_heads=6,
        feed_forward_dim=512,
        num_transformer_blocks=6,
        rngs=nnx.Rngs(0),
    )

    print(model)
