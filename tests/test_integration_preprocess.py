from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from datetime import datetime
import time
import cv2
import imageio
from PIL import Image

from core.configuration import Config
from core.loader import Loader
from core.dataset_manager import DatasetManager
from preprocess.cropping import Cropping
from preprocess.resampler import IsotropicResampler
from preprocess.windowing import Windowing

# ==========================================================
# CONFIGURATION DES CAS DE TEST
# ==========================================================

TEST_CASES = [
    (10011, 220375232),  # Patient 10011 - L-CC
    (10011, 270344397),  # Patient 10011 - L-MLO
    (10011, 541722628),  # Patient 10011 - R-CC
    (10011, 1031443799),  # Patient 10011 - R-MLO
    (10130, 613462606),  # Patient 10130 - L-CC
]

# Configuration de la sauvegarde
SAVE_STEP_IMAGES = True  # Sauvegarder les images à chaque étape
VISUALIZE_ALL_IMAGES = True  # Générer les visualisations détaillées
VISUALIZE_SUMMARY = True  # Générer le résumé visuel global


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture(scope="module")
def setup_pipeline(tmp_path_factory):
    """Initialize pipeline components."""
    csv_path = Path("../data/train.csv")
    images_dir = Path("../data/train_images")
    out_dir = tmp_path_factory.mktemp("resampled")

    config = Config(csv_path=csv_path, images_dir=images_dir, out_dir=out_dir)
    loader = Loader(config)
    dataset_manager = DatasetManager(config, loader)
    cropping = Cropping(config, loader, dataset_manager)
    windowing = Windowing(preserve_range=(0.0, 1.0))
    resampler = IsotropicResampler(
        out_dir=out_dir,
        target_nominal=0.10,
        max_pixels=100_000_000,
        upsample_max=2.0,
        downsample_max=3.0
    )

    return dataset_manager, cropping, windowing, resampler


# ==========================================================
# HELPER FUNCTIONS - SAUVEGARDE DES IMAGES
# ==========================================================

def save_step_image(image: np.ndarray, step_name: str, output_dir: Path,
                    patient_id: int, image_id: int, cmap: str = 'gray'):
    """Sauvegarde une image à une étape spécifique du pipeline"""
    step_dir = output_dir / "step_images" / f"patient_{patient_id}_image_{image_id}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Normaliser l'image pour la sauvegarde
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.min() < 0 or image.max() > 1:
            # Normaliser vers [0, 1]
            img_normalized = (image - image.min()) / (image.max() - image.min())
        else:
            img_normalized = image
    else:
        img_normalized = image.astype(np.float32) / image.max()

    # Sauvegarder en PNG
    output_path = step_dir / f"{step_name}.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(img_normalized, cmap=cmap)
    plt.axis('off')
    plt.title(f'{step_name}\nShape: {image.shape}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Sauvegarder en NPY pour les données brutes
    npy_path = step_dir / f"{step_name}.npy"
    np.save(npy_path, image)

    print(f"     💾 Saved {step_name}: {output_path}")
    return output_path


def save_comparison_grid(images_dict: Dict[str, np.ndarray], output_dir: Path,
                         patient_id: int, image_id: int, metadata: Dict):
    """Crée une grille de comparaison de toutes les étapes"""
    grid_dir = output_dir / "comparison_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    steps = list(images_dict.keys())
    for i, (step_name, image) in enumerate(images_dict.items()):
        if i >= len(axes):
            break

        # Normaliser l'image pour l'affichage
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.min() < 0 or image.max() > 1:
                img_display = (image - image.min()) / (image.max() - image.min())
            else:
                img_display = image
        else:
            img_display = image.astype(np.float32) / image.max()

        axes[i].imshow(img_display, cmap='gray')
        axes[i].set_title(f'{step_name}\n{image.shape}', fontsize=10, fontweight='bold')
        axes[i].axis('off')

        # Ajouter des statistiques
        stats_text = f"Min: {image.min():.1f}\nMax: {image.max():.1f}\nMean: {image.mean():.2f}"
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Cacher les axes non utilisés
    for i in range(len(steps), len(axes)):
        axes[i].axis('off')

    # Titre principal
    view_info = f"{metadata.get('laterality', '?')}-{metadata.get('view', '?')}"
    plt.suptitle(f'Patient {patient_id}, Image {image_id} ({view_info})\nPipeline Step Comparison',
                 fontsize=16, fontweight='bold', y=0.95)

    output_path = grid_dir / f"comparison_p{patient_id}_i{image_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"     🔄 Comparison grid: {output_path}")
    return output_path


def compute_histogram_entropy(image: np.ndarray, bins: int = 50) -> float:
    """Compute normalized histogram entropy."""
    hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 1))
    hist = hist.astype(np.float64)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(len(hist)) if len(hist) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def visualize_single_image(
        patient_id: int,
        image_id: int,
        original: np.ndarray,
        cropped: np.ndarray,
        windowed: np.ndarray,
        resampled: np.ndarray,
        metadata: Dict,
        results: Dict,
        output_dir: Path
):
    """Generate comprehensive visualization for a single image."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # 1. Original DICOM
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(original, cmap='gray', vmin=original.min(), vmax=np.percentile(original, 99))
    ax1.set_title(f'Original DICOM\nShape: {original.shape}\nSpacing: {results["original_spacing"]} mm/px',
                  fontsize=10, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. After Cropping
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(cropped, cmap='gray')
    ax2.set_title(f'After Cropping\nShape: {cropped.shape}\nRange: [{cropped.min():.0f}, {cropped.max():.0f}]',
                  fontsize=10, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. After Windowing
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(windowed, cmap='gray', vmin=0, vmax=1)
    density_text = f"Density: {metadata.get('density') or 'B (default)'}"
    ax3.set_title(f'After Windowing\n{density_text}\nStd: {results["windowed_std"]:.3f}',
                  fontsize=10, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. After Resampling
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(resampled, cmap='gray', vmin=0, vmax=1)
    ax4.set_title(
        f'After Resampling\nShape: {resampled.shape}\nSpacing: {results["resampled_spacing"]} mm/px\nPolicy: {results["resampled_policy"]}',
        fontsize=10, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Original histogram
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(original.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax5.set_title(f'Original Histogram\nMean: {original.mean():.1f}', fontweight='bold', fontsize=10)
    ax5.set_xlabel('Intensity')
    ax5.set_ylabel('Density')
    ax5.grid(alpha=0.3)
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 6. Cropped histogram
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(cropped.flatten(), bins=100, alpha=0.7, color='green', edgecolor='black', density=True)
    ax6.set_title(f'Cropped Histogram\nMean: {cropped.mean():.1f}', fontweight='bold', fontsize=10)
    ax6.set_xlabel('Intensity')
    ax6.set_ylabel('Density')
    ax6.grid(alpha=0.3)
    ax6.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 7. Windowed histogram
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(windowed.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black', density=True)
    ax7.set_title(f'Windowed Histogram\nMean: {windowed.mean():.3f}, Std: {windowed.std():.3f}',
                  fontweight='bold', fontsize=10)
    ax7.set_xlabel('Intensity [0-1]')
    ax7.set_ylabel('Density')
    ax7.set_xlim(0, 1)
    ax7.grid(alpha=0.3)

    # 8. Resampled histogram
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(resampled.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black', density=True)
    ax8.set_title(f'Resampled Histogram\nMean: {resampled.mean():.3f}, Std: {resampled.std():.3f}',
                  fontweight='bold', fontsize=10)
    ax8.set_xlabel('Intensity [0-1]')
    ax8.set_ylabel('Density')
    ax8.set_xlim(0, 1)
    ax8.grid(alpha=0.3)

    # Main title
    view_info = f"{metadata.get('laterality', '?')}-{metadata.get('view', '?')}"
    cancer_info = "CANCER" if metadata.get('cancer') == 1 else "Normal"
    plt.suptitle(
        f'Patient {patient_id}, Image {image_id} ({view_info}) - {cancer_info} | Time: {results["time_total_s"]:.2f}s',
        fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_path = output_dir / "detailed_visualizations" / f"patient_{patient_id}_image_{image_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"     📊 Detailed viz: {output_path}")


def process_single_image(
        patient_id: int,
        image_id: int,
        dataset_manager: DatasetManager,
        cropping: Cropping,
        windowing: Windowing,
        resampler: IsotropicResampler,
        output_dir: Path
) -> Dict:
    """
    Process a single image through the full pipeline and save all step images.
    """
    results = {
        'patient_id': patient_id,
        'image_id': image_id,
        'success': False,
        'error': None
    }

    start_time = time.time()
    step_images = {}  # Pour stocker les images de chaque étape

    try:
        # Get metadata
        dicom_info = dataset_manager.get_dicom_info(patient_id, image_id)
        metadata = {
            'view': dicom_info.get('view'),
            'laterality': dicom_info.get('laterality'),
            'density': dicom_info.get('density'),
            'cancer': dicom_info.get('cancer'),
            'age': dicom_info.get('age')
        }
        results.update(metadata)

        # ÉTAPE 1: Chargement DICOM original
        print("     📥 Loading DICOM...")
        dicom_path = dataset_manager.get_dicom_path(patient_id, image_id)
        dicom_data = dataset_manager.dicom_record(Path(dicom_path), verbose=False)
        image_original = dicom_data["image"]
        spacing = dicom_data["spacing"]

        if SAVE_STEP_IMAGES:
            save_step_image(image_original, "01_original_dicom", output_dir, patient_id, image_id)
        step_images["01_original"] = image_original

        load_time = time.time() - start_time

        results['original_shape'] = image_original.shape
        results['original_spacing'] = spacing
        results['original_dtype'] = str(image_original.dtype)
        results['original_size_mb'] = image_original.nbytes / (1024 ** 2)

        # ÉTAPE 2: Cropping
        print("     ✂️  Cropping...")
        crop_start = time.time()
        crop_result = cropping.process_one(
            patient_id=patient_id,
            image_id=image_id,
            laterality=metadata.get('laterality'),
            view=metadata.get('view'),
            dicom_path=dicom_path
        )

        image_cropped = crop_result['raw_crop']
        image_crop_model = crop_result['crop_model']

        if SAVE_STEP_IMAGES:
            save_step_image(image_cropped, "02_cropped_raw", output_dir, patient_id, image_id)
            save_step_image(image_crop_model, "03_cropped_normalized", output_dir, patient_id, image_id)
        step_images["02_cropped_raw"] = image_cropped
        step_images["03_cropped_norm"] = image_crop_model

        crop_time = time.time() - crop_start

        results['cropped_shape'] = image_cropped.shape
        results['cropped_range'] = (float(image_cropped.min()), float(image_cropped.max()))
        results['bbox'] = crop_result['bbox']
        results['flipped'] = crop_result['flipped']

        # ÉTAPE 3: Windowing
        print("     🪟 Windowing...")
        window_start = time.time()
        image_windowed = windowing.process_one(
            image_crop_model,
            density=metadata.get('density')
        )

        if SAVE_STEP_IMAGES:
            save_step_image(image_windowed, "04_windowed", output_dir, patient_id, image_id)
        step_images["04_windowed"] = image_windowed

        window_time = time.time() - window_start

        results['windowed_shape'] = image_windowed.shape
        results['windowed_mean'] = float(image_windowed.mean())
        results['windowed_std'] = float(image_windowed.std())
        results['windowed_min'] = float(image_windowed.min())
        results['windowed_max'] = float(image_windowed.max())
        results['windowed_entropy'] = float(compute_histogram_entropy(image_windowed))

        # ÉTAPE 4: Resampling
        print("     📐 Resampling...")
        resample_start = time.time()
        stem = f"{patient_id}_{image_id}"
        resample_result = resampler.process_one(
            stem=stem,
            img_np=image_windowed,
            spacing=spacing
        )

        # Load resampled image
        image_resampled = np.load(resample_result["npy"])["img"]

        if SAVE_STEP_IMAGES:
            save_step_image(image_resampled, "05_resampled", output_dir, patient_id, image_id)
        step_images["05_resampled"] = image_resampled

        resample_time = time.time() - resample_start

        results['resampled_shape'] = image_resampled.shape
        results['resampled_spacing'] = resample_result['new_spacing_mm']
        results['resampled_policy'] = resample_result['policy_reason']
        results['resampled_mean'] = float(image_resampled.mean())
        results['resampled_std'] = float(image_resampled.std())
        results['resampled_dtype'] = str(image_resampled.dtype)
        results['resampled_size_mb'] = image_resampled.nbytes / (1024 ** 2)

        # Créer la grille de comparaison
        if SAVE_STEP_IMAGES:
            save_comparison_grid(step_images, output_dir, patient_id, image_id, metadata)

        # Validation
        results['has_nan'] = bool(np.isnan(image_resampled).any())
        results['has_inf'] = bool(np.isinf(image_resampled).any())
        results['std_is_valid'] = not (np.isnan(image_resampled.std()) or np.isinf(image_resampled.std()))

        # Timing
        total_time = time.time() - start_time
        results['time_load_s'] = load_time
        results['time_crop_s'] = crop_time
        results['time_window_s'] = window_time
        results['time_resample_s'] = resample_time
        results['time_total_s'] = total_time

        # Visualisation détaillée
        if VISUALIZE_ALL_IMAGES:
            visualize_single_image(
                patient_id, image_id,
                image_original, image_cropped, image_windowed, image_resampled,
                metadata, results, output_dir
            )

        results['success'] = True
        print("     ✅ Pipeline completed successfully!")

    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        print(f"     ❌ Error: {e}")

    return results


# ==========================================================
# MAIN TEST
# ==========================================================

def test_pipeline_custom_patients(setup_pipeline):
    """
    Test the full pipeline on custom-defined patients and images.
    """
    dataset_manager, cropping, windowing, resampler = setup_pipeline

    print(f"\n{'=' * 70}")
    print(f"🚀 CUSTOM MULTI-PATIENT PIPELINE TEST")
    print(f"{'=' * 70}")
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Step images saving: {SAVE_STEP_IMAGES}")
    print(f"Detailed visualizations: {VISUALIZE_ALL_IMAGES}")
    print(f"Summary report: {VISUALIZE_SUMMARY}")
    print(f"{'=' * 70}\n")

    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("pipeline_output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")

    # Process all images
    all_results = []

    for i, (patient_id, image_id) in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] Processing Patient {patient_id}, Image {image_id}")
        print(f"{'─' * 70}")

        result = process_single_image(
            patient_id=patient_id,
            image_id=image_id,
            dataset_manager=dataset_manager,
            cropping=cropping,
            windowing=windowing,
            resampler=resampler,
            output_dir=output_dir
        )

        all_results.append(result)

        if result['success']:
            print(f"     ✅ Success!")
            print(f"     📊 Original: {result['original_shape']} @ {result['original_spacing']} mm/px")
            print(f"     📊 Cropped: {result['cropped_shape']} (flipped: {result.get('flipped', False)})")
            print(f"     📊 Windowed: Mean={result['windowed_mean']:.3f}, Std={result['windowed_std']:.3f}")
            print(f"     📊 Resampled: {result['resampled_shape']} @ {result['resampled_spacing']} mm/px")
            print(f"     ⏱️  Time: {result['time_total_s']:.2f}s")

    # Save results to CSV
    df_results = pd.DataFrame(all_results)
    csv_path = output_dir / "pipeline_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"📁 Results CSV saved: {csv_path}")

    # Save summary JSON
    successful = [r for r in all_results if r['success']]
    summary = {
        'timestamp': timestamp,
        'total_images': len(all_results),
        'successful': len(successful),
        'failed': len(all_results) - len(successful),
        'success_rate_pct': len(successful) / len(all_results) * 100 if all_results else 0,
        'avg_processing_time_s': np.mean([r['time_total_s'] for r in successful]) if successful else 0,
        'total_processing_time_s': sum([r['time_total_s'] for r in successful]) if successful else 0,
        'output_directory': str(output_dir),
    }

    json_path = output_dir / "pipeline_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Summary JSON saved: {json_path}")

    # Print final summary
    print(f"\n{'=' * 70}")
    print(f"📈 FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"✅ Success: {len(successful)}/{len(all_results)} ({summary['success_rate_pct']:.1f}%)")
    print(f"⏱️  Total time: {summary['total_processing_time_s']:.2f}s")
    print(f"📁 All outputs saved in: {output_dir}")

    # Assertions pour le test
    assert len(successful) > 0, "Aucune image n'a été traitée avec succès"
    assert summary['success_rate_pct'] >= 50.0, f"Taux de succès trop bas: {summary['success_rate_pct']:.1f}%"


if __name__ == "__main__":
    # Pour exécuter le test directement
    pytest.main([__file__, "-v"])