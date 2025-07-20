import importlib
import sys
import types

def test_train_real_data_entry():
    mod = importlib.import_module('scripts.train_real_data')
    assert hasattr(mod, 'main')

def test_evaluate_real_entry():
    mod = importlib.import_module('scripts.evaluate_real')
    assert hasattr(mod, 'main')

def test_inference_real_entry():
    mod = importlib.import_module('scripts.inference_real')
    assert hasattr(mod, 'main') 