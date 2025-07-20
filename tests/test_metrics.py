import torch
from src.metrics import EvaluationSuite

def test_evaluation_suite():
    sample_rate = 16000
    evaluator = EvaluationSuite(sample_rate)
    batch = {
        'clean': torch.randn(2, sample_rate * 2),
        'noisy': torch.randn(2, sample_rate * 2)
    }
    class DummyModel:
        def eval(self):
            pass
        def __call__(self, noisy_audio, mode='inference'):
            return {'enhanced_audio': noisy_audio}
    model = DummyModel()
    results = evaluator.evaluate_batch(model, batch)
    assert 'pesq' in results
    assert 'stoi' in results
    assert 'si_sdr' in results
    assert 'snr' in results 