import pydicom
import matplotlib.pyplot as plt

def show_two_dicoms(dicom_path1, dicom_path2, title1="Image 1", title2="Image 2"):
    # Lire les DICOM
    ds1 = pydicom.dcmread(dicom_path1)
    ds2 = pydicom.dcmread(dicom_path2)

    img1 = ds1.pixel_array
    img2 = ds2.pixel_array

    # Affichage côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.show()

# Exemple d'utilisation
dicom1 = "/Users/assadiabira/Bureau/Kaggle/Projet_kaggle/data/train/45_1476454372.dcm"
dicom2 = "/Users/assadiabira/Bureau/Kaggle/Projet_kaggle/tests/dicom_output/train/45_1476454372.dcm"
show_two_dicoms(dicom1, dicom2)
