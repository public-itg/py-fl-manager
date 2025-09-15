import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageChops

from fl_manager.components.formatters.pillow import ImagePathToBytesFormatter


@pytest.fixture(scope='module')
def image_dataframe():
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)

    _image_path = []
    for i in range(10):
        image_path = tmpdir_path / f'image_{i}.png'
        im_array = np.random.rand(28, 28) * 255
        im = Image.fromarray(im_array.astype('uint8')).convert('L')
        im.save(image_path)
        _image_path.append(image_path)

    data = {'image_path': _image_path, 'label': range(10)}
    yield pd.DataFrame(data)
    tmpdir.cleanup()


def test_image_path_to_bytes_formatter(image_dataframe):
    formatter = ImagePathToBytesFormatter(
        in_col_name='image_path', out_col_name='image_bytes'
    )
    r = formatter.run(image_dataframe)
    assert 'image_bytes' in r.columns
    _sample = r.iloc[0]
    assert isinstance(_sample.image_bytes, bytes)
    image_from_path = Image.open(_sample.image_path)
    image_from_bytes = Image.open(io.BytesIO(_sample.image_bytes))
    diff = ImageChops.difference(image_from_path, image_from_bytes)
    assert not diff.getbbox()
