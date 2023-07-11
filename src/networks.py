import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=input_embedding_size)
        self.norm2 = nn.LayerNorm(input_embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.silu = nn.SiLU()

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear1(x)
        ff_output = self.linear2(ff_output)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        x = self.silu(x)
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

        for param in self.input_embedding.parameters():
            param.requires_grad = False

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

        for param in self.input_embedding.parameters():
            param.requires_grad = False

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
