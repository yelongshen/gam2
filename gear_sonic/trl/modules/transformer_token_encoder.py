from __future__ import annotations

import torch
from torch import nn


class TransformerTokenEncoder(nn.Module):
    """Temporal Transformer encoder that emits universal-token latents.

    Input tensors follow the same contract as the existing MLP encoders:
    ``(batch, num_input_temporal_dims, input_dim)``. The output follows the
    UniversalTokenModule encoder contract: ``(batch, num_output_temporal_dims,
    output_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_input_temporal_dims: int,
        num_output_temporal_dims: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_input_temporal_dims = num_input_temporal_dims
        self.num_output_temporal_dims = num_output_temporal_dims

        self.input_proj = nn.Linear(input_dim, d_model)
        self.frame_pos_embed = nn.Parameter(
            torch.zeros(1, num_input_temporal_dims, d_model)
        )
        self.token_queries = nn.Parameter(
            torch.zeros(1, num_output_temporal_dims, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.frame_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_queries, std=0.02)

    def forward(self, input, frame_mask=None, **kwargs):
        if input.ndim == 2:
            input = input.view(-1, self.num_input_temporal_dims, self.input_dim)

        frame_tokens = self.input_proj(input) + self.frame_pos_embed[:, : input.shape[-2]]
        query_tokens = self.token_queries.expand(input.shape[0], -1, -1)
        tokens = torch.cat([frame_tokens, query_tokens], dim=-2)

        key_padding_mask = None
        if frame_mask is not None:
            query_mask = torch.ones(
                frame_mask.shape[0],
                self.num_output_temporal_dims,
                dtype=torch.bool,
                device=frame_mask.device,
            )
            valid_mask = torch.cat([frame_mask, query_mask], dim=-1)
            key_padding_mask = ~valid_mask

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        query_output = encoded[:, -self.num_output_temporal_dims :]
        return self.output_proj(self.output_norm(query_output))