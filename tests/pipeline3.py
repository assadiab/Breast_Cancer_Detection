from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import datetime
import torch
from tqdm import tqdm

from core.configuration import Config
from core.loader import Loader
from core.dataset_manager import DatasetManager
from preprocess.cropping import Cropping
from preprocess.windowing import Windowing
from preprocess.resampler import IsotropicResampler

OUTPUT_DIR = Path("dicom_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ==========================================================
# SAVE DICOM (UNCOMPRESSED)
# ==========================================================
def save_as_dicom_uncompressed(image: np.ndarray, original_dicom: Path, output_path: Path):
    """Save a proper, uncompressed DICOM, even if original was compressed."""
    ds_orig = pydicom.dcmread(original_dicom)
    file_meta = ds_orig.file_meta
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0"*128)

    # Copy metadata except PixelData
    for elem in ds_orig:
        if elem.tag.group != 0x7FE0:
            ds.add(elem)

    # Normalize image to 0-65535 and convert to uint16
    img_scaled = np.clip(image, 0, 1)
    img_uint16 = (img_scaled * 65535).astype(np.uint16)

    ds.Rows, ds.Columns = img_uint16.shape
    ds.PixelData = img_uint16.tobytes()
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    # Force uncompressed transfer syntax
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Update creation date/time
    dt = datetime.datetime.now()
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S")

    ds.save_as(output_path)

# ==========================================================
# PIPELINE GPU + BATCH
# ==========================================================
def run_pipeline_gpu_batch(batch_size=16):
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "has_mps") and torch.has_mps and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Environment: {device}, CPUs: {torch.get_num_threads()}")

    # Load CSVs
    CSV_PATHS = {
        'test': Path("../data/csv/X_test.csv")
    }
    IMAGE_PATHS = {
        'test': Path("../data/test/")
    }
    LABEL_PATHS = {
        'test': Path("../data/csv/y_test.csv")
    }

    splits = {}
    for split in ['test']:
        X_df = pd.read_csv(CSV_PATHS[split])
        y_df = pd.read_csv(LABEL_PATHS[split])
        merged = pd.merge(X_df, y_df, on=['patient_id','image_id']) if 'image_id' in y_df.columns else X_df.copy()
        if 'cancer' not in merged.columns:
            merged['cancer'] = y_df.iloc[:,0]
        splits[split] = merged
        print(f"✅ Loaded {split}: {len(merged)} samples")

    # Initialize pipeline components
    config = Config(csv_path=CSV_PATHS['test'], images_dir=IMAGE_PATHS['test'], out_dir=OUTPUT_DIR)
    loader = Loader(config)
    dataset_manager = DatasetManager(config, loader)
    cropping = Cropping(config, loader, dataset_manager)
    windowing = Windowing(preserve_range=(0.0,1.0))
    resampler = IsotropicResampler(out_dir=OUTPUT_DIR)

    # ======================================================
    # Helper: resample batch on GPU
    # ======================================================
    def resample_batch(imgs, spacings):
        """Resample a batch of images using isotropic resampling."""
        resampled = []
        for img, sp in zip(imgs, spacings):
            img_res, _ = resampler.resample_isotropic(img, sp)
            resampled.append(img_res)
        return resampled

    # ======================================================
    # Process each split
    # ======================================================
    for split_name, df in splits.items():
        print(f"\nProcessing split: {split_name}, {len(df)} images")
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(exist_ok=True)

        batch_imgs, batch_spacings, batch_files = [], [], []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name} images"):
            patient_id, image_id = row['patient_id'], row['image_id']
            # Find DICOM
            dicom_file = None
            for f in IMAGE_PATHS[split_name].glob(f"*{image_id}*.dcm"):
                dicom_file = f
                break
            if dicom_file is None:
                continue

            # Load DICOM image & info
            dicom_info = dataset_manager.get_dicom_info(patient_id,image_id)
            dicom_data = dataset_manager.dicom_record(dicom_file, verbose=False)
            img = dicom_data['image']
            spacing = dicom_data['spacing']

            # Crop + window
            img_crop = cropping.process_one(patient_id,image_id,dicom_info['laterality'],dicom_info['view'],dicom_file)['crop_model']
            img_window = windowing.process_one(img_crop,density=dicom_info.get('density'))

            # Append to batch
            batch_imgs.append(img_window)
            batch_spacings.append(spacing)
            batch_files.append((patient_id, image_id, dicom_file))

            # If batch full, process
            if len(batch_imgs) == batch_size:
                imgs_res = resample_batch(batch_imgs, batch_spacings)
                for im_res, (pid, iid, dicom_file) in zip(imgs_res, batch_files):
                    out_file = split_dir / f"{pid}_{iid}.dcm"
                    save_as_dicom_uncompressed(im_res, dicom_file, out_file)
                batch_imgs, batch_spacings, batch_files = [], [], []

        # Process remaining batch
        if batch_imgs:
            imgs_res = resample_batch(batch_imgs, batch_spacings)
            for im_res, (pid, iid, dicom_file) in zip(imgs_res, batch_files):
                out_file = split_dir / f"{pid}_{iid}.dcm"
                save_as_dicom_uncompressed(im_res, dicom_file, out_file)

# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    run_pipeline_gpu_batch(batch_size=16)
