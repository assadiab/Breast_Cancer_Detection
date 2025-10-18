import pytest
import numpy as np
from pathlib import Path
from core.loader import Loader
from core.configuration import Config
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import datetime

# ---------------- Fixtures CSV ---------------- #
@pytest.fixture
def tmp_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    data = pd.DataFrame({
        'patient_id': [1, 2],
        'image_id': [10, 20],
        'density': ['A', 'B']
    })
    data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def tmp_images_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return str(images_dir)

@pytest.fixture
def config(tmp_csv, tmp_images_dir):
    return Config(csv_path=tmp_csv, images_dir=tmp_images_dir)

# ---------------- Helper DICOM ---------------- #
def create_dummy_dicom(path: Path, pixel_array: np.ndarray):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = 'CT'
    ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')
    ds.Rows, ds.Columns = pixel_array.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 2.0
    ds.RescaleIntercept = 10.0
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(str(path))

# ---------------- Tests ---------------- #
def test_load_dataframe(config):
    loader = Loader(config)
    loader.load_df()
    df = loader.get_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2

def test_load_dicom(tmp_images_dir, config):
    dicom_path = Path(tmp_images_dir) / "dummy.dcm"
    arr = np.array([[0, 50], [100, 200]], dtype=np.uint16)
    create_dummy_dicom(dicom_path, arr)

    loader = Loader(config)
    img = loader.load_dicom(str(dicom_path))
    # Check slope/intercept applied
    assert img[0,0] == arr[0,0]*2 + 10
    assert img[1,1] == arr[1,1]*2 + 10
    assert img.shape == arr.shape

def test_load_multiple_dicoms(tmp_images_dir, config):
    dicom_paths = []
    for i in range(2):
        path = Path(tmp_images_dir) / f"dummy_{i}.dcm"
        arr = np.array([[i*10, i*20], [i*30, i*40]], dtype=np.uint16)
        create_dummy_dicom(path, arr)
        dicom_paths.append(str(path))

    loader = Loader(config)
    images = loader.load_multiple_dicoms(dicom_paths)
    assert len(images) == 2
    for i, img in enumerate(images):
        assert img[0,0] == i*10*2 + 10
