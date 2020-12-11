import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """

        : embed_size    => shape of the embeddings
        : heads         => split count of the embeddings, I believe it is 8
                            if you have embed size of 256, then it is 8*32 heads

    """

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * self.heads == self.embed_size
        ), "Embed size need to be divisible by heads"

        """ 
            *** create the keys, value and queries from the embeddings *** 
        """

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(
            self.head_dim * self.heads, embed_size
        )  # you could write embed size times embed size

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]  # no of many examples we send in at a time
        value_len, keys_len, queries_len = (
            values.shape[1],
            keys.shape[1],
            queries.shape[1],
        )  # they gonna mostly depend upon the inputs and target length

        # Split the embeddimgs into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len) => query_len: target length, key_len: src length

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # apply a softmax in the energy, which is a combination of Q,V,K
        # here we want to apply the softmax in the source(keys) to find a relationship
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # attention shape: (N, heads, query_len, keys_len)
        # values shape: (N, value_len, heads, head_dim)
        # output shape: (N, query_len, heads, head_dim)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, queries_len, self.heads * self.head_dim
        )

        # after einsum reshape or flatten the last two dimension by multiplying them
        out = self.fc_out(out)
        return out


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.layer_norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.layer_norm2(forward + x))
        return out


# Encoder Block in which the inputs are processed
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


# Decoder
class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)

        out = self.fc_out(x)
        return out


# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        src_pad_idx,
        target_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            N, 1, target_len, target_len
        )
        return target_mask.to(self.device)

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)
        return out


# transformer completed #


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(
        device
    )

    src_pad_idx = 0
    target_pad_idx = 0
    src_vocab_size = 10
    target_vocab_size = 10

    model = Transformer(
        src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx, device=device
    ).to(device)

    out = model(x, target[:, :-1])
    print(out.shape)
