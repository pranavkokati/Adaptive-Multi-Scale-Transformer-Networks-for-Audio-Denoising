import pytest
from src.real_datasets import RealDatasetManager
from src.librispeech_dataset import LibriSpeechRealDataset
from pathlib import Path

@pytest.mark.parametrize("dataset_name", ["voicebank_demand", "microsoft_dns"])
def test_real_dataset_loading(dataset_name, tmp_path):
    # Use a temp path, but this will only test instantiation, not actual download
    manager = RealDatasetManager(tmp_path, sample_rate=16000, segment_length=4.0)
    # Should not raise error on instantiation
    assert manager is not None


def test_librispeech_dataset_loading(tmp_path):
    # Only test instantiation, not download
    dataset = LibriSpeechRealDataset(tmp_path, sample_rate=16000, segment_length=4.0)
    assert dataset is not None 