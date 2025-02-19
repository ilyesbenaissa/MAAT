import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class CoGatedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CoGatedAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x, x_1):
        attn_output, attn_weights = self.attention(x, x_1, x_1)
        gated_output = torch.sigmoid(self.gate(torch.cat((x, attn_output), dim=-1)))
        return gated_output, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None, d_model=512):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension d_model
            d_state=8,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to("cuda")
        self.gate = nn.Linear(d_model * 2, d_model)  # Gate for 0-1 gating

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        original_x = x  # Save the original input for the skip connection

        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            x_skip = self.mamba(x) + original_x  # Apply skip connection here
            x_skip = self.norm(x_skip)
            gate = torch.sigmoid(self.gate(torch.cat((x, x_skip), dim=-1)))  # 0-1 gating
            x = gate * x_skip + (1 - gate) * x  # Apply gating

            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            original_x = x  # Update original_x for the next layer

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class MambaAnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, block_size=10, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(MambaAnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, d_model, n_heads, block_size, attention_dropout=dropout,
                                         output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            d_model=d_model
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
