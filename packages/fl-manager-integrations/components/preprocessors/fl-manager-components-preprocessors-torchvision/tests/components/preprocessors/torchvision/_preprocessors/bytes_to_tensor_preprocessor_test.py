import io

import numpy as np
import torch
from PIL import Image

from fl_manager.components.preprocessors.torchvision import BytesToTensorPreprocessor


def test_bytes_to_tensor_preprocessor():
    preprocessor = BytesToTensorPreprocessor()
    _bytes = io.BytesIO()
    im_array = np.random.rand(28, 28) * 255
    im = Image.fromarray(im_array.astype('uint8')).convert('L')
    im.save(_bytes, format='png')
    _tensor = preprocessor.preprocess(_bytes.getvalue())
    assert isinstance(_tensor, torch.Tensor)
    assert _tensor.shape[1:] == (28, 28)
