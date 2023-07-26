import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from torch import Tensor
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimplePositionalEncoding(nn.Module):
    def __init__(self, num_channels, input_embedding_size):
        super(SimplePositionalEncoding, self).__init__()
        self.pos_encoding = torch.zeros(num_channels, input_embedding_size)
        position = torch.arange(0, num_channels, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_embedding_size, 2).float() * (
                -torch.log(torch.tensor(10000.0)) / input_embedding_size))
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = self.pos_encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1)].to(x.device)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_channels, input_embedding_size):
        super(LearnedPositionalEncoding, self).__init__()
        self.positional_encoding = nn.Embedding(num_channels, input_embedding_size)

    def forward(self, x):
        batch_size, seq_len, input_embedding_size = x.size()
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        pos_encoding = self.positional_encoding(positions)

        return x + pos_encoding


class TransformerBlock(nn.Module):
    def __init__(self, input_embedding_size, num_heads, hidden_size, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_embedding_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_embedding_size)
        self.linear1 = nn.Linear(in_features=input_embedding_size, out_features=hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=input_embedding_size)
        self.norm2 = nn.LayerNorm(input_embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear1(x)
        ff_output = self.relu1(ff_output)
        ff_output = self.linear2(ff_output)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_embedding_size, dropout):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 20), stride=(1, 1)),
            nn.BatchNorm2d(num_features=40),
            nn.SiLU(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(22, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=40),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 15)),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=40, out_channels=input_embedding_size, kernel_size=(1, 1), stride=(1, 1)),
            nn.Flatten(start_dim=2)
        )

    def forward(self, x):
        return self.conv_block(x)


class ShallProjBlock(nn.Module):
    def __init__(self, input_embedding_size, dropout):
        super(ShallProjBlock, self).__init__()

        self.shallow_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 10), stride=(1, 1)),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(22, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 15)),
            nn.Dropout(dropout),
        )

        self.projection_net = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=input_embedding_size, kernel_size=(1, 1), stride=(1, 1)),
            nn.Flatten(start_dim=2),
        )

    def forward(self, x):
        x = self.shallow_net(x)
        x = self.projection_net(x)
        return x


class EEGTransformerBasic(nn.Module):
    def __init__(self,
                 num_layers,
                 num_channels,
                 num_heads,
                 window_size,
                 input_embedding_size,
                 hidden_size,
                 dropout,
                 do_positional_encoding,
                 learned_positional_encoding,
                 num_classes):
        super(EEGTransformerBasic, self).__init__()
        self.do_positional_encoding = do_positional_encoding
        self.learned_positional_encoding = learned_positional_encoding
        self.input_embedding = nn.Linear(in_features=window_size, out_features=input_embedding_size)
        self.simple_pos_encoding = SimplePositionalEncoding(num_channels=num_channels,
                                                            input_embedding_size=input_embedding_size)
        self.learned_pos_encoding = LearnedPositionalEncoding(num_channels=num_channels,
                                                              input_embedding_size=input_embedding_size)
        self.encoder = nn.ModuleList(
            [TransformerBlock(input_embedding_size=input_embedding_size,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              dropout=dropout) for _ in range(num_layers)])
        self.ff = nn.Sequential(
            nn.Linear(in_features=input_embedding_size, out_features=512),
            nn.SiLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.SiLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.SiLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.input_embedding(x)
        if self.do_positional_encoding:
            if self.learned_positional_encoding:
                x = self.learned_pos_encoding(x)
            else:
                x = self.simple_pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        x = torch.mean(input=x, dim=1)
        x = self.ff(x)
        return x


class EEGTransformerInverse(nn.Module):
    def __init__(self,
                 num_layers,
                 num_channels,
                 num_heads,
                 window_size,
                 input_embedding_size,
                 hidden_size,
                 dropout,
                 do_positional_encoding,
                 learned_positional_encoding,
                 num_classes):
        super(EEGTransformerInverse, self).__init__()
        self.do_positional_encoding = do_positional_encoding
        self.learned_positional_encoding = learned_positional_encoding
        self.input_embedding = nn.Linear(in_features=num_channels, out_features=input_embedding_size)
        self.simple_pos_encoding = SimplePositionalEncoding(num_channels=window_size,
                                                            input_embedding_size=input_embedding_size)
        self.learned_pos_encoding = LearnedPositionalEncoding(num_channels=window_size,
                                                              input_embedding_size=input_embedding_size)
        self.encoder = nn.ModuleList(
            [TransformerBlock(input_embedding_size=input_embedding_size,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              dropout=dropout) for _ in range(num_layers)])
        self.ff = nn.Sequential(
            nn.Linear(in_features=input_embedding_size, out_features=512),
            nn.SiLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.SiLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.SiLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        )

        # for param in self.input_embedding.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = torch.transpose(input=x, dim0=1, dim1=2)
        x = self.input_embedding(x)
        if self.do_positional_encoding:
            if self.learned_positional_encoding:
                x = self.learned_pos_encoding(x)
            else:
                x = self.simple_pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.ff(x)
        return x


class EEGTransformerConv(nn.Module):
    def __init__(self,
                 num_layers,
                 num_channels,
                 num_heads,
                 window_size,
                 input_embedding_size,
                 hidden_size,
                 dropout,
                 do_positional_encoding,
                 learned_positional_encoding,
                 num_classes):
        super(EEGTransformerConv, self).__init__()
        self.conv_net = ConvBlock(input_embedding_size, dropout)
        self.encoder = nn.ModuleList(
            [TransformerBlock(input_embedding_size=input_embedding_size,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_embedding_size),
            nn.Linear(in_features=input_embedding_size, out_features=num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv_net(x)
        x = torch.transpose(input=x, dim0=1, dim1=2)
        for layer in self.encoder:
            x = layer(x)
        x = torch.mean(input=x, dim=1)
        x = self.fc(x)
        return x


class EEGTransformerConvFFT(nn.Module):
    def __init__(self,
                 num_layers,
                 num_channels,
                 num_heads,
                 window_size,
                 input_embedding_size,
                 hidden_size,
                 dropout,
                 do_positional_encoding,
                 learned_positional_encoding,
                 num_classes):
        super(EEGTransformerConvFFT, self).__init__()
        self.conv_net = ConvBlock(input_embedding_size, dropout)
        self.encoder = nn.ModuleList(
            [TransformerBlock(input_embedding_size=input_embedding_size,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_embedding_size),
            nn.Linear(in_features=input_embedding_size, out_features=num_classes),
        )

    def forward(self, x):
        x = (torch.fft.fft(x, dim=-1)).to(device=x.device, dtype=torch.float)
        x = x.unsqueeze(dim=1)
        x = self.conv_net(x)
        x = torch.transpose(input=x, dim0=1, dim1=2)
        for layer in self.encoder:
            x = layer(x)
        x = torch.mean(input=x, dim=1)
        x = self.fc(x)
        return x


class EEGTransformerMaster(nn.Module):
    def __init__(self,
                 num_layers,
                 num_channels,
                 num_heads,
                 window_size,
                 input_embedding_size,
                 hidden_size,
                 dropout,
                 do_positional_encoding,
                 learned_positional_encoding,
                 num_classes):
        super(EEGTransformerMaster, self).__init__()
        self.shallow_projection_net = ShallProjBlock(input_embedding_size, dropout)
        self.encoder = nn.ModuleList(
            [TransformerBlock(input_embedding_size=1104,
                              num_heads=num_heads,
                              hidden_size=hidden_size,
                              dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.LayerNorm(normalized_shape=1104),
            nn.Linear(in_features=1104, out_features=2440),
            nn.Linear(in_features=2440, out_features=256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256, out_features=32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=32, out_features=num_classes)
        )

    def forward(self, x):
        x = (torch.fft.fft(x, dim=-1)).to(device=x.device, dtype=torch.float)
        x = torch.transpose(input=x, dim0=1, dim1=2)
        x = x.unsqueeze(dim=1)
        x = self.shallow_projection_net(x)
        for layer in self.encoder:
            x = layer(x)
        x = torch.mean(input=x, dim=1)
        x = self.fc(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        # x = torch.transpose(input=x, dim0=2, dim1=3)
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, n_heads, dropout):
        super().__init__(*[TransformerEncoderBlock(emb_size, n_heads, dropout) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(emb_size * 69, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_heads=10, n_classes=4, dropout=0.5, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size, n_heads, dropout),
            ClassificationHead(emb_size, n_classes)
        )