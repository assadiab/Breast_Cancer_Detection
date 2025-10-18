import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess.resampler import IsotropicResampler

@pytest.fixture
def tmp_out_dir(tmp_path):
    out_dir = tmp_path / "resampled"
    out_dir.mkdir()
    return out_dir

@pytest.fixture
def sample_images():
    """
    Create a list of sample images with spacing and patient_id
    matching the expected structure of the real pipeline.
    """
    images = []
    for i in range(3):
        img_np = np.random.randint(0, 2000, size=(84, 74), dtype=np.int16)
        images.append({
            'stem': f'img_{i}',
            'image': img_np,
            'spacing': (0.27, 0.27),
            'patient_id': i,
            'image_id': i  # <- important pour process_all_chunks
        })
    return images

def test_process_all_chunks(tmp_out_dir, sample_images):
    # Création du DataFrame correspondant à la vraie structure
    df = pd.DataFrame(sample_images)

    resampler = IsotropicResampler(out_dir=tmp_out_dir, target_nominal=0.2, chunk_size=2)
    df_res = resampler.process_all_chunks(df)

    # Vérifications
    assert isinstance(df_res, pd.DataFrame)
    # Colonnes présentes dans la sortie réelle
    expected_cols = {'image_id', 'stem', 'patient_id', 'orig_spacing_mm', 'new_spacing_mm', 'npy', 'json', 'shape', 'policy_reason'}
    assert set(df_res.columns) >= expected_cols
    assert len(df_res) == len(sample_images)

    for idx, row in df_res.iterrows():
        # Vérifie que chaque image a été traitée
        assert 'image_id' in row
        assert row['image_id'] == row['image_id']  # correspondance id
        assert row['orig_spacing_mm'] is not None
        assert row['new_spacing_mm'] is not None
        assert row['npy'] is not None
        assert row['json'] is not None
