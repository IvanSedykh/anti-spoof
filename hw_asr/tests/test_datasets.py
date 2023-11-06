import torch

from hw_asr.datasets import LibrispeechMixDataset
from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser


def test_librispeech_mix():
    config_parser = ConfigParser.get_test_configs()
    with clear_log_folder_after_use(config_parser):
        ds = LibrispeechMixDataset(
            "data/datasets/librispeech/dev-clean",
            config_parser=config_parser,
            max_audio_length=13,
            limit=10,
        )
        assert len(ds) == 10

        item = ds[0]
        assert isinstance(item["ref_wav"], torch.Tensor)
        assert isinstance(item["mix_wav"], torch.Tensor)
        assert isinstance(item["target_wav"], torch.Tensor)
        assert isinstance(item["speaker_id"], torch.Tensor)
