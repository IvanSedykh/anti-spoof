import torch
from torch import nn
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=9, padding=4)
        # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=1, padding=0)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True
        )
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # todo:check mask arg
        # print(f"{non_pad_mask.shape=}")
        # [bs, seq_len, 1]
        # print(f"{slf_attn_mask.shape=}")
        # [bs, seq_len, seq_len]

        # [bs, seq_len, seq_len] -> [bs*num_heads, seq_len, seq_len]
        slf_attn_mask = slf_attn_mask.repeat(self.slf_attn.num_heads, 1, 1)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask
        )

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class DurationPredictor(nn.Module):
    """Duration Predictor"""

    def __init__(self, model_config):
        super().__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size, kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count + k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, model_config):
        super().__init__()
        self.duration_predictor = DurationPredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(torch.sum(duration_predictor_output, -1), -1)[
            0
        ].item()
        alignment = torch.zeros(
            duration_predictor_output.size(0),
            expand_max_len,
            duration_predictor_output.size(1),
        ).numpy()
        alignment = create_alignment(alignment, duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is not None:
            output = self.LR(target, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha
            ).int()
            output = self.LR(x, duration_predictor_output)
            # wtf is this
            mel_pos = (
                torch.stack([torch.Tensor([i + 1 for i in range(output.size(1))])])
                .long()
                .to(output.device)
            )
            return output, mel_pos


def get_non_pad_mask(seq, model_config):
    assert seq.dim() == 2
    return seq.ne(model_config.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, model_config):
    """For masking out the padding part of key sequence."""
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(model_config.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config

        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.position_enc = nn.Embedding(
            n_position, model_config.encoder_dim, padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    model_config.encoder_dim,
                    model_config.encoder_conv1d_filter_size,
                    model_config.encoder_head,
                    dropout=model_config.dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(
            seq_k=src_seq, seq_q=src_seq, model_config=self.model_config
        )
        non_pad_mask = get_non_pad_mask(src_seq, model_config=self.model_config)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """Decoder"""

    def __init__(self, model_config):
        super().__init__()

        len_max_seq = model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer
        self.model_config = model_config

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    model_config.encoder_dim,
                    model_config.encoder_conv1d_filter_size,
                    model_config.encoder_head,
                    dropout=model_config.dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, enc_pos, return_attns=False):
        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(
            seq_k=enc_pos, seq_q=enc_pos, model_config=self.model_config
        )
        non_pad_mask = get_non_pad_mask(enc_pos, model_config=self.model_config)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech(nn.Module):
    """FastSpeech"""

    def __init__(self, model_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.0)

    def forward(
        self,
        src_seq,
        src_pos,
        mel_pos=None,
        mel_max_length=None,
        length_target=None,
        alpha=1.0,
    ):
        encoder_output, _ = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(
                encoder_output,
                target=length_target,
                alpha=alpha,
                mel_max_length=mel_max_length,
            )
            decoder_output = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)

            return mel_output, duration_predictor_output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(
                encoder_output, alpha=alpha
            )

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)

            return mel_output
