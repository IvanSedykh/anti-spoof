from dataclasses import dataclass
import pytest
import torch

from src.model.fastspeech import Decoder, DurationPredictor, Encoder, FastSpeech, LengthRegulator, PositionwiseFeedForward, FFTBlock



@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000
    num_mels = 80

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = "<blank>"
    UNK_WORD = "<unk>"
    BOS_WORD = "<s>"
    EOS_WORD = "</s>"


@dataclass
class TrainConfig:
    checkpoint_path = "./model_new"
    logger_path = "./logger"
    mel_ground_truth = "./data/mels"
    alignment_path = "./data/alignments"
    data_path = "./data/train.txt"

    wandb_project = "fastspeech_example"

    text_cleaners = ["english_cleaners"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"

    batch_size = 10
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 3000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32


@pytest.fixture
def train_config():
    return TrainConfig()


@pytest.fixture
def model_config():
    return FastSpeechConfig()


def test_pff():
    din = 64
    dhid = 128

    layer = PositionwiseFeedForward(d_in=din, d_hid=dhid)

    inp = torch.randn(12, 128, din)

    out = layer(inp)

    assert inp.shape == out.shape


def test_fftblock():
    dmod = 64
    d_inner = 128
    seq_len = 90
    bs = 12

    layer = FFTBlock(d_model=dmod, d_inner=d_inner, n_head=4)

    inp = torch.randn(bs, seq_len, dmod)

    out, attn_scores = layer(inp)
    assert inp.shape == out.shape
    assert attn_scores.shape == (bs, seq_len, seq_len)


def test_duration_predictor(model_config: FastSpeechConfig):
    layer = DurationPredictor(model_config)

    seq_len = 90
    bs = 12

    inp = torch.randn(bs, seq_len, model_config.encoder_dim)
    out = layer(inp)

    assert out.shape == (bs, seq_len)


def test_LenghtRegulator(model_config: FastSpeechConfig):
    layer = LengthRegulator(model_config)

    bs = 12
    seq_len = 30
    inp = torch.randn(bs, seq_len, model_config.encoder_dim)

    # todo: check
    out, mel_pos = layer(inp)
    # assert out.shape == (bs, mel_pos.max(), model_config.encoder_dim)
    # assert mel_pos.shape == (bs,)

def test_Encoder(model_config: FastSpeechConfig):
    encoder = Encoder(model_config)

    bs = 12
    seq_len = 30

    inp = torch.randint(0, model_config.vocab_size, (bs, seq_len))
    src_pos = torch.arange(seq_len).unsqueeze(0).repeat(bs, 1)
    enc_out, non_pad_mask = encoder(inp, src_pos)

    assert enc_out.shape == (bs, seq_len, model_config.encoder_dim)
    assert non_pad_mask.shape == (bs, seq_len, 1)


def test_Decoder(model_config: FastSpeechConfig):
    decoder = Decoder(model_config)

    bs = 12
    seq_len = 30

    enc_out = torch.randn(bs, seq_len, model_config.encoder_dim)
    src_pos = torch.arange(seq_len).unsqueeze(0).repeat(bs, 1)
    dec_output = decoder(enc_out, src_pos)
    assert dec_output.shape == (bs, seq_len, model_config.decoder_dim)


def test_FastSpeech(model_config: FastSpeechConfig):
    model = FastSpeech(model_config)

    bs = 12
    seq_len = 30

    inp = torch.randint(0, model_config.vocab_size, (bs, seq_len))
    src_pos = torch.arange(seq_len).unsqueeze(0).repeat(bs, 1)
