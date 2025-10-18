import torch
import numpy as np
from preprocess.windowing import Windowing


def test_forward_with_density():
    model = Windowing()
    dummy_image = torch.rand(1, 1, 512, 512)  # batch=1, grayscale
    result = model(dummy_image, density='B')

    assert 'windowed_image' in result
    assert 'params' in result
    windowed = result['windowed_image']
    assert windowed.min() >= 0.0 and windowed.max() <= 1.0


def test_forward_without_density():
    model = Windowing()
    dummy_image = torch.rand(1, 1, 512, 512)
    result = model(dummy_image, density=None)

    assert 'windowed_image' in result
    assert 'params' in result
    windowed = result['windowed_image']
    assert windowed.min() >= 0.0 and windowed.max() <= 1.0


def test_fallback_methods():
    model = Windowing()
    dummy_image = np.random.rand(512, 512).astype(np.float32)

    for density in ['A', 'B', 'C', 'D']:
        output = model.apply_fallback(dummy_image, density)
        assert isinstance(output, np.ndarray)
        assert output.min() >= 0.0 and output.max() <= 1.0
