from pathlib import Path
import pytest
import numpy as np
import matplotlib.pyplot as plt
from core.configuration import Config
from core.loader import Loader
from core.dataset_manager import DatasetManager
from preprocess.cropping import Cropping
from preprocess.resampler import IsotropicResampler
from preprocess.windowing import Windowing


@pytest.fixture
def setup_pipeline(tmp_path):
    """
    Initialize Config, Loader, DatasetManager and preprocessing objects.
    """
    csv_path = Path("../data/train.csv")
    images_dir = Path("../data/train_images")
    out_dir = tmp_path

    config = Config(csv_path=csv_path, images_dir=images_dir, out_dir=out_dir)
    loader = Loader(config)
    dataset_manager = DatasetManager(config, loader)

    cropping = Cropping()
    windowing = Windowing(preserve_range=(0.0, 1.0))  # Mode adaptatif par défaut
    resampler = IsotropicResampler(out_dir)

    return dataset_manager, cropping, windowing, resampler


@pytest.mark.parametrize("patient_id,image_id", [
    (10011, 220375232),
])
def test_full_pipeline(setup_pipeline, patient_id, image_id):
    """Test du pipeline complet de préprocessing."""
    dataset_manager, cropping, windowing, resampler = setup_pipeline

    # ==========================================================
    # DEBUG ÉTAPE 0: VÉRIFICATION DES MÉTADONNÉES
    # ==========================================================
    print(f"\n{'=' * 60}")
    print(f"🔍 DEBUG ÉTAPE 0 - EXTRACTION DES MÉTADONNÉES")
    print(f"{'=' * 60}")

    # Récupération des métadonnées
    dicom_info = dataset_manager.get_dicom_info(patient_id, image_id)
    density = dicom_info.get('density', None)

    # Debug complet des métadonnées disponibles
    print("📋 Métadonnées disponibles:")
    for key, value in dicom_info.items():
        if value is not None and key != 'dicom_path':  # Éviter le chemin trop long
            print(f"   {key}: {value}")

    print(f"🎯 Densité extraite: {repr(density)}")
    print(f"   Type: {type(density)}")

    # Validation de la densité
    if density and isinstance(density, str) and density.upper() in ['A', 'B', 'C', 'D']:
        density = density.upper()
        print(f"✅ Densité valide: {density}")
    else:
        print(f"⚠️  Densité non valide ou manquante, utilisation du défaut (B)")
        density = None

    # Load DICOM
    dicom_path = dataset_manager.get_dicom_path(patient_id, image_id)
    dicom_data = dataset_manager.dicom_record(Path(dicom_path), verbose=True)
    image = dicom_data["image"]
    spacing = dicom_data["spacing"]

    print(f"\n{'=' * 60}")
    print(f"Testing patient {patient_id}, image {image_id}")
    if density:
        print(f"Breast density: {density}")
    else:
        print(f"Breast density: Not specified (using default B)")
    print(f"{'=' * 60}\n")

    # ==========================================================
    # 1. TESTS - IMAGE ORIGINALE
    # ==========================================================
    assert isinstance(image, np.ndarray), "Image must be numpy array"
    assert image.ndim == 2, "Image must be 2D"
    assert np.isfinite(image).all(), "Image contains non-finite values"
    print(f"✓ Original image: {image.shape}, dtype={image.dtype}")

    # ==========================================================
    # 2. TESTS - CROPPING
    # ==========================================================
    print(f"\n{'=' * 60}")
    print(f"🔍 DEBUG ÉTAPE 2 - CROPPING")
    print(f"{'=' * 60}")

    image_cropped = cropping.process_with_metadata(image)

    assert isinstance(image_cropped, np.ndarray)
    assert image_cropped.shape == (512, 512), f"Expected (512, 512), got {image_cropped.shape}"
    assert np.isfinite(image_cropped).all()
    assert image_cropped.shape[0] <= image.shape[0]
    assert image_cropped.shape[1] <= image.shape[1]

    print(f"✓ Cropped image: {image_cropped.shape}")
    print(f"  Range: [{image_cropped.min():.1f}, {image_cropped.max():.1f}]")

    # ==========================================================
    # 3. TESTS - WINDOWING (ÉTAPE CRITIQUE)
    # ==========================================================
    print(f"\n{'=' * 60}")
    print(f"🔍 DEBUG ÉTAPE 3 - WINDOWING ADAPTATIF")
    print(f"{'=' * 60}")
    print(f"📤 Passage de la densité au windowing: {repr(density)}")

    # DEBUG: Vérifier les paramètres qui seront utilisés
    if density and density in windowing.density_params:
        params = windowing.density_params[density]
        print(f"⚙️  Paramètres pour la densité {density}:")
        print(f"   - Percentiles: {params['percentiles']}")
        print(f"   - Gamma: {params['gamma']}")
        print(f"   - CLAHE clip: {params['clahe_clip']}")
        print(f"   - CLAHE weight: {params['clahe_weight']}")
    else:
        default_params = windowing.density_params['B']
        print(f"⚙️  Paramètres par défaut (densité B):")
        print(f"   - Percentiles: {default_params['percentiles']}")
        print(f"   - Gamma: {default_params['gamma']}")
        print(f"   - CLAHE clip: {default_params['clahe_clip']}")
        print(f"   - CLAHE weight: {default_params['clahe_weight']}")

    image_windowed = windowing.process_one(image_cropped, density=density)

    assert isinstance(image_windowed, np.ndarray)
    assert image_windowed.shape == image_cropped.shape, "Shape must not change"
    assert np.isfinite(image_windowed).all()
    assert image_windowed.min() >= 0.0, f"Min value {image_windowed.min()} < 0"
    assert image_windowed.max() <= 1.0, f"Max value {image_windowed.max()} > 1"

    # Vérifier la qualité du windowing
    windowed_std = image_windowed.std()
    assert windowed_std > 0.05, f"Windowing std too low: {windowed_std:.3f}"

    # Vérifier la diversité de l'histogramme
    histogram_entropy = compute_histogram_entropy(image_windowed)
    assert histogram_entropy > 0.3, f"Histogram too concentrated: {histogram_entropy:.3f}"

    print(f"✓ Windowed image: {image_windowed.shape}")
    print(f"  Range: [{image_windowed.min():.3f}, {image_windowed.max():.3f}]")
    print(f"  Mean: {image_windowed.mean():.3f}, Std: {windowed_std:.3f}")
    print(f"  Histogram entropy: {histogram_entropy:.3f}")

    # ==========================================================
    # 4. TESTS - RESAMPLING
    # ==========================================================
    print(f"\n{'=' * 60}")
    print(f"🔍 DEBUG ÉTAPE 4 - RESAMPLING")
    print(f"{'=' * 60}")

    stem = f"{patient_id}_{image_id}"
    result = resampler.process_one(stem=stem, img_np=image_windowed, spacing=spacing)

    assert isinstance(result, dict)
    assert "npy" in result and "json" in result and "shape" in result

    image_resampled = np.load(result["npy"])["img"]

    print(f"\n{'=' * 60}")
    print(f"🔍 VALIDATION COMPLÈTE IMAGE RESAMPLED")
    print(f"{'=' * 60}")

    # 1. Vérifications de base
    print(f"📊 Propriétés de base:")
    print(f"   Shape: {image_resampled.shape}")
    print(f"   Dtype: {image_resampled.dtype}")
    print(f"   Size: {image_resampled.nbytes / 1024:.1f} KB")

    # 2. Statistiques
    print(f"\n📈 Statistiques:")
    print(f"   Min: {image_resampled.min():.6f}")
    print(f"   Max: {image_resampled.max():.6f}")
    print(f"   Mean: {image_resampled.mean():.6f}")
    print(f"   Std: {image_resampled.std():.6f}")
    print(f"   Median: {np.median(image_resampled):.6f}")

    # 3. Valeurs invalides
    print(f"\n⚠️  Vérification valeurs invalides:")
    num_nan = np.isnan(image_resampled).sum()
    num_inf = np.isinf(image_resampled).sum()
    num_neg = (image_resampled < 0).sum()
    num_over = (image_resampled > 1).sum()

    print(f"   NaN: {num_nan} ({num_nan / image_resampled.size * 100:.2f}%)")
    print(f"   Inf: {num_inf} ({num_inf / image_resampled.size * 100:.2f}%)")
    print(f"   Négatifs: {num_neg} ({num_neg / image_resampled.size * 100:.2f}%)")
    print(f"   > 1.0: {num_over} ({num_over / image_resampled.size * 100:.2f}%)")

    # 4. Distribution
    print(f"\n📊 Distribution:")
    print(f"   Valeurs uniques: {len(np.unique(image_resampled))}")
    print(f"   Percentiles:")
    for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        val = np.percentile(image_resampled, p)
        print(f"      P{p:3d}: {val:.6f}")

    # 5. Comparaison avec windowed
    print(f"\n🔄 Comparaison Windowed vs Resampled:")
    print(f"   Windowed  - Mean: {image_windowed.mean():.6f}, Std: {image_windowed.std():.6f}")
    print(f"   Resampled - Mean: {image_resampled.mean():.6f}, Std: {image_resampled.std():.6f}")
    print(f"   Différence - Mean: {abs(image_windowed.mean() - image_resampled.mean()):.6f}")
    print(f"   Différence - Std:  {abs(image_windowed.std() - image_resampled.std()):.6f}")

    # 6. Test de l'histogramme
    print(f"\n📊 Test calcul histogramme:")
    try:
        hist, bins = np.histogram(image_resampled.flatten(), bins=50, range=(0, 1))
        print(f"   Histogramme calculé avec succès")
        print(f"   Bins non-vides: {(hist > 0).sum()}")
        print(f"   Total counts: {hist.sum()}")
    except Exception as e:
        print(f"   ❌ ERREUR: {e}")

    # 7. Test du std
    print(f"\n🧮 Test calcul std:")
    try:
        std_numpy = np.std(image_resampled)
        std_nanstd = np.nanstd(image_resampled)
        print(f"   np.std():    {std_numpy}")
        print(f"   np.nanstd(): {std_nanstd}")
        print(f"   Sont égaux: {np.isclose(std_numpy, std_nanstd)}")

        if np.isinf(std_numpy):
            print(f"   ❌ STD EST INF !")
        elif np.isnan(std_numpy):
            print(f"   ❌ STD EST NAN !")
        else:
            print(f"   ✅ STD est valide")
    except Exception as e:
        print(f"   ❌ ERREUR lors du calcul: {e}")

    print(f"{'=' * 60}\n")

    # Assertions strictes
    assert not np.isnan(image_resampled).any(), "Resampled contains NaN"
    assert not np.isinf(image_resampled).any(), "Resampled contains Inf"
    assert not np.isinf(image_resampled.std()), "Resampled std is Inf"
    assert not np.isnan(image_resampled.std()), "Resampled std is NaN"


    # ==========================================================
    # 5. RÉSUMÉ DU PIPELINE
    # ==========================================================
    print(f"\n{'=' * 60}")
    print(f"🎯 RÉSUMÉ FINAL DU PIPELINE")
    print(f"{'=' * 60}")
    print(f"Densité utilisée: {density or 'B (défaut)'}")
    print(f"Original    : {image.shape} → range [{image.min():.1f}, {image.max():.1f}]")
    print(f"Cropped     : {image_cropped.shape} → range [{image_cropped.min():.1f}, {image_cropped.max():.1f}]")
    print(f"Windowed    : {image_windowed.shape} → range [{image_windowed.min():.3f}, {image_windowed.max():.3f}]")
    print(f"Resampled   : {image_resampled.shape} → spacing {result['new_spacing_mm']} mm")
    print(f"{'=' * 60}\n")

    # ==========================================================
    # 6. VISUALISATIONS CORRIGÉES
    # ==========================================================
    visualize_pipeline_corrected(
        patient_id, image_id,
        image, image_cropped, image_windowed, image_resampled,
        density
    )

    print(f"✅ Pipeline validation complete!")


def compute_histogram_entropy(image: np.ndarray, bins: int = 50) -> float:
    """
    Mesure la diversité de l'histogramme (entropie normalisée).
    Valeur élevée = bonne distribution des intensités.
    Retourne une valeur entre 0 (concentré) et 1 (uniforme).
    """
    # Calculer l'histogramme (counts bruts, pas density)
    hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 1))

    # Normaliser pour obtenir des probabilités
    hist = hist.astype(np.float64)
    hist = hist / hist.sum()

    # Filtrer les bins vides
    hist = hist[hist > 0]

    # Calculer l'entropie de Shannon (toujours positive)
    entropy = -np.sum(hist * np.log2(hist))

    # Normaliser par l'entropie maximale (log2 du nombre de bins non-vides)
    max_entropy = np.log2(len(hist)) if len(hist) > 1 else 1.0

    return entropy / max_entropy if max_entropy > 0 else 0.0


def visualize_pipeline_corrected(
        patient_id: int,
        image_id: int,
        original: np.ndarray,
        cropped: np.ndarray,
        windowed: np.ndarray,
        resampled: np.ndarray,
        density: str
):
    """
    Crée des visualisations complètes du pipeline - VERSION CORRIGÉE.
    """
    vis_dir = Path("test_visualizations")
    vis_dir.mkdir(exist_ok=True)

    # ==========================================
    # FIGURE 1: Comparaison côte à côte
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Original
    im0 = axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title(f'Original DICOM\nShape: {original.shape}', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Cropped
    im1 = axes[0, 1].imshow(cropped, cmap='gray')
    axes[0, 1].set_title(f'After Cropping\nShape: {cropped.shape}', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Windowed
    im2 = axes[1, 0].imshow(windowed, cmap='gray')
    title = f'After Windowing (Adaptive)\nRange: [{windowed.min():.3f}, {windowed.max():.3f}]'
    if density:
        title += f'\nDensity: {density}'
    else:
        title += f'\nDensity: B (default)'
    axes[1, 0].set_title(title, fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Resampled
    im3 = axes[1, 1].imshow(resampled, cmap='gray')
    axes[1, 1].set_title(f'After Resampling\nShape: {resampled.shape}', fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(f'Patient {patient_id}, Image {image_id} - Full Pipeline',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(vis_dir / f"{patient_id}_{image_id}_pipeline.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {vis_dir / f'{patient_id}_{image_id}_pipeline.png'}")

    # ==========================================
    # FIGURE 2: HISTOGRAMMES CORRIGÉS
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calcul des bins adaptatifs pour chaque image
    bins_original = 100
    bins_normalized = 50  # Moins de bins pour les images normalisées [0,1]

    # HISTOGRAMME ORIGINAL
    axes[0, 0].hist(original.flatten(), bins=bins_original, alpha=0.7,
                    color='blue', edgecolor='black', density=True)
    axes[0, 0].set_title(f"Original DICOM Histogram\nMean: {original.mean():.1f}, Std: {original.std():.1f}",
                         fontweight='bold', fontsize=11)
    axes[0, 0].set_xlabel("Intensity (HU/Units)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # HISTOGRAMME CROPPED
    axes[0, 1].hist(cropped.flatten(), bins=bins_original, alpha=0.7,
                    color='green', edgecolor='black', density=True)
    axes[0, 1].set_title(f"Cropped Histogram\nMean: {cropped.mean():.1f}, Std: {cropped.std():.1f}",
                         fontweight='bold', fontsize=11)
    axes[0, 1].set_xlabel("Intensity (HU/Units)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # HISTOGRAMME WINDOWED
    axes[1, 0].hist(windowed.flatten(), bins=bins_normalized, alpha=0.7,
                    color='red', edgecolor='black', density=True)
    axes[1, 0].set_title(f"Windowed Histogram\nMean: {windowed.mean():.3f}, Std: {windowed.std():.3f}",
                         fontweight='bold', fontsize=11)
    axes[1, 0].set_xlabel("Intensity [0-1]")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim(0, 1)  # Fixer l'échelle pour comparaison

    # HISTOGRAMME RESAMPLED
    axes[1, 1].hist(resampled.flatten(), bins=bins_normalized, alpha=0.7,
                    color='purple', edgecolor='black', density=True)
    axes[1, 1].set_title(f"Resampled Histogram\nMean: {resampled.mean():.3f}, Std: {resampled.std():.3f}",
                         fontweight='bold', fontsize=11)
    axes[1, 1].set_xlabel("Intensity [0-1]")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim(0, 1)  # Fixer l'échelle pour comparaison

    plt.suptitle(f'Patient {patient_id}, Image {image_id} - Histograms Analysis (Corrected)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / f"{patient_id}_{image_id}_histograms.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {vis_dir / f'{patient_id}_{image_id}_histograms.png'}")

    # ==========================================
    # FIGURE 3: PROFILS D'INTENSITÉ CORRIGÉS
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # PROFIL HORIZONTAL - Windowed (512 points)
    center_row_w = windowed.shape[0] // 2
    x_windowed = np.arange(windowed.shape[1])
    axes[0, 0].plot(x_windowed, windowed[center_row_w, :],
                    label='Windowed', linewidth=2, color='red')
    axes[0, 0].set_title("Horizontal Profile - Windowed Image", fontweight='bold')
    axes[0, 0].set_xlabel("Column Index (0-511)")
    axes[0, 0].set_ylabel("Intensity")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # PROFIL HORIZONTAL - Resampled (171 points)
    center_row_r = resampled.shape[0] // 2
    x_resampled = np.arange(resampled.shape[1])
    axes[0, 1].plot(x_resampled, resampled[center_row_r, :],
                    label='Resampled', linewidth=2, color='purple')
    axes[0, 1].set_title("Horizontal Profile - Resampled Image", fontweight='bold')
    axes[0, 1].set_xlabel("Column Index (0-170)")
    axes[0, 1].set_ylabel("Intensity")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # PROFIL VERTICAL - Windowed (512 points)
    center_col_w = windowed.shape[1] // 2
    y_windowed = np.arange(windowed.shape[0])
    axes[1, 0].plot(y_windowed, windowed[:, center_col_w],
                    label='Windowed', linewidth=2, color='red')
    axes[1, 0].set_title("Vertical Profile - Windowed Image", fontweight='bold')
    axes[1, 0].set_xlabel("Row Index (0-511)")
    axes[1, 0].set_ylabel("Intensity")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # PROFIL VERTICAL - Resampled (171 points)
    center_col_r = resampled.shape[1] // 2
    y_resampled = np.arange(resampled.shape[0])
    axes[1, 1].plot(y_resampled, resampled[:, center_col_r],
                    label='Resampled', linewidth=2, color='purple')
    axes[1, 1].set_title("Vertical Profile - Resampled Image", fontweight='bold')
    axes[1, 1].set_xlabel("Row Index (0-170)")
    axes[1, 1].set_ylabel("Intensity")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.suptitle(f'Patient {patient_id}, Image {image_id} - Intensity Profiles (Corrected)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / f"{patient_id}_{image_id}_profiles.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {vis_dir / f'{patient_id}_{image_id}_profiles.png'}")


# Test pour plusieurs densités
@pytest.mark.parametrize("density", ["A", "B", "C", "D", None])
def test_windowing_densities(setup_pipeline, density):
    """Test le windowing avec différentes densités."""
    dataset_manager, cropping, windowing, _ = setup_pipeline

    patient_id, image_id = 10011, 220375232
    dicom_path = dataset_manager.get_dicom_path(patient_id, image_id)
    dicom_data = dataset_manager.dicom_record(Path(dicom_path), verbose=False)

    image_cropped = cropping.process_with_metadata(dicom_data["image"])

    print(f"\n🧪 TEST WINDOWING MANUEL - Densité: {repr(density)}")
    image_windowed = windowing.process_one(image_cropped, density=density)

    # Assertions
    assert image_windowed.min() >= 0.0 and image_windowed.max() <= 1.0
    assert image_windowed.std() > 0.05, f"Density {density}: std too low"

    entropy = compute_histogram_entropy(image_windowed)
    assert entropy > 0.3, f"Density {density}: histogram too concentrated"

    print(f"📊 Density {density or 'None'}: std={image_windowed.std():.3f}, entropy={entropy:.3f}")