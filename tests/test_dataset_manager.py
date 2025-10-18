from unittest.mock import MagicMock
import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import datetime
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from core.dataset_manager import DatasetManager
from core.configuration import Config
from core.loader import Loader

# Mock Config with single images_dir
class MockConfig:
    images_dir = "/images"

# Sample DataFrame
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "patient_id": [1, 2],
        "image_id": [10, 20],
        "density": ["A", "C"],
        "laterality": ["L", "R"],
        "view": ["CC", "MLO"],
        "age": [50, 60],
        "cancer": [0, 1],
        "biopsy": [0, 1],
        "invasive": [0, 1],
        "BIRADS": ["2", "4"],
        "implant": [0, 0],
        "difficult_negative_case": [False, True]
    })

@pytest.fixture
def dataset_manager(sample_df):
    loader_mock = MagicMock(spec=Loader)
    loader_mock.get_df.return_value = sample_df
    return DatasetManager(config=MockConfig(), loader=loader_mock)

def test_get_dicom_path(dataset_manager):
    path = dataset_manager.get_dicom_path(patient_id=1, image_id=10)
    assert path == "/images/1/10.dcm"

def test_get_dicom_info_existing(dataset_manager):
    info = dataset_manager.get_dicom_info(patient_id=2, image_id=20)
    expected_keys = list(dataset_manager.loader.get_df().columns) + ['dicom_path', 'patient_id', 'image_id']
    assert all(k in info for k in expected_keys)
    assert info['patient_id'] == 2
    assert info['image_id'] == 20
    assert info['dicom_path'] == "/images/2/20.dcm"
    # Check that values match CSV
    df_row = dataset_manager.loader.get_df().loc[
        (dataset_manager.loader.get_df()['patient_id'] == 2) &
        (dataset_manager.loader.get_df()['image_id'] == 20)
    ].iloc[0]
    for col in dataset_manager.loader.get_df().columns:
        assert info[col] == df_row[col]

def test_get_dicom_info_missing(dataset_manager):
    info = dataset_manager.get_dicom_info(patient_id=999, image_id=999)
    # Toutes les colonnes CSV sauf patient_id/image_id doivent être None
    for col in dataset_manager.loader.get_df().columns:
        if col not in ['patient_id', 'image_id']:
            assert info[col] is None
    # dicom_path, patient_id, image_id doivent être présents
    assert info['dicom_path'] == "/images/999/999.dcm"
    assert info['patient_id'] == 999
    assert info['image_id'] == 999

# ---------------- Fixture Config ---------------- #
@pytest.fixture
def config(tmp_path):
    csv_file = tmp_path / "dummy.csv"
    csv_file.write_text("patient_id,image_id\n1,10")  # 1 ligne suffit
    return Config(csv_path=str(csv_file), images_dir=str(tmp_path))

# ---------------- Helper pour créer un DICOM ---------------- #
def create_dummy_dicom(path: Path, pixel_array: np.ndarray):
    """Crée un DICOM temporaire valide avec pixel_array"""
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
    ds.PixelData = pixel_array.tobytes()
    ds.PixelSpacing = [0.2, 0.2]

    ds.save_as(str(path))

# ---------------- Test dicom_record ---------------- #
def test_dicom_record(tmp_path, config):
    # Créer un dossier images/patient
    patient_dir = tmp_path / "1"
    patient_dir.mkdir()
    dicom_path = patient_dir / "10.dcm"

    # Créer image 2x2
    pixel_array = np.array([[0, 50], [100, 200]], dtype=np.uint16)
    create_dummy_dicom(dicom_path, pixel_array)

    # Initialiser loader et dataset_manager
    loader = Loader(config)
    dataset_manager = DatasetManager(config=config, loader=loader)

    # Appel de dicom_record
    record = dataset_manager.dicom_record(dicom_path)

    # Assertions
    assert record["path"] == str(dicom_path)
    assert record["spacing"] == (0.2, 0.2)
    assert "image" in record
    assert record["image"].shape == pixel_array.shape
