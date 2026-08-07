from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


FLASH_ATTN_MAX_BATCH_HEADS = 65535


class _FlashTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
    ):
        super().__init__()
        try:
            from flash_attn import flash_attn_qkvpacked_func
        except ImportError as exc:
            raise ImportError(
                "flash-attn is required when use_flash_attn=True"
            ) from exc

        self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.norm_first = norm_first

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation for flash attention: {activation}")

    def forward(self, src, src_key_padding_mask=None):
        if src_key_padding_mask is not None:
            raise NotImplementedError("use_flash_attn=True does not support frame_mask yet")
        if not src.is_cuda:
            raise RuntimeError("use_flash_attn=True requires CUDA tensors")

        if self.norm_first:
            src = src + self._sa_block(self.norm1(src))
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(self, src):
        batch_size, seq_len, _ = src.shape
        qkv = self.qkv_proj(src).view(
            batch_size, seq_len, 3, self.nhead, self.head_dim
        )
        flash_dtype = src.dtype if src.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        attn_output = self.flash_attn_qkvpacked_func(
            qkv.to(flash_dtype),
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model).to(src.dtype)
        return self.dropout1(self.out_proj(attn_output))

    def _ff_block(self, src):
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(src))))
        )


class _FlashTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _FlashTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        return output


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
        max_encoder_batch_size: int = 8192,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_input_temporal_dims = num_input_temporal_dims
        self.num_output_temporal_dims = num_output_temporal_dims
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            flash_max_batch_size = FLASH_ATTN_MAX_BATCH_HEADS // nhead
            self.max_encoder_batch_size = min(max_encoder_batch_size, flash_max_batch_size)
        else:
            self.max_encoder_batch_size = max_encoder_batch_size

        self.input_proj = nn.Linear(input_dim, d_model)
        self.frame_pos_embed = nn.Parameter(
            torch.zeros(1, num_input_temporal_dims, d_model)
        )
        self.token_queries = nn.Parameter(
            torch.zeros(1, num_output_temporal_dims, d_model)
        )

        if use_flash_attn:
            self.encoder = _FlashTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
        else:
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
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.frame_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_queries, std=0.02)
        for layer in self.encoder.layers:
            if hasattr(layer, "self_attn"):
                nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
                if layer.self_attn.in_proj_bias is not None:
                    nn.init.zeros_(layer.self_attn.in_proj_bias)
                nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
                nn.init.zeros_(layer.self_attn.out_proj.bias)
            else:
                nn.init.xavier_uniform_(layer.qkv_proj.weight)
                nn.init.zeros_(layer.qkv_proj.bias)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                nn.init.zeros_(layer.out_proj.bias)

            nn.init.kaiming_uniform_(layer.linear1.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.linear1.bias)
            nn.init.kaiming_uniform_(layer.linear2.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.linear2.bias)

            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

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

        if tokens.shape[0] > self.max_encoder_batch_size:
            encoded_chunks = []
            for start in range(0, tokens.shape[0], self.max_encoder_batch_size):
                end = start + self.max_encoder_batch_size
                chunk_key_padding_mask = (
                    key_padding_mask[start:end] if key_padding_mask is not None else None
                )
                token_chunk = tokens[start:end]
                if self.training and torch.is_grad_enabled():
                    encoded_chunks.append(
                        checkpoint(
                            lambda chunk: self.encoder(
                                chunk,
                                src_key_padding_mask=chunk_key_padding_mask,
                            ),
                            token_chunk,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    )
                else:
                    encoded_chunks.append(
                        self.encoder(
                            token_chunk,
                            src_key_padding_mask=chunk_key_padding_mask,
                        )
                    )
            encoded = torch.cat(encoded_chunks, dim=0)
        else:
            encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        query_output = encoded[:, -self.num_output_temporal_dims :]
        return self.output_proj(self.output_norm(query_output))