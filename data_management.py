import os
from pathlib import Path
import cv2

def extract_label_from_filename(filename: str) -> int:
    """
    Extreu el número guanyador a partir del nom del fitxer seguint el format:
    [tipus]_[id]_[numero].png

    Parameters
    ----------
    filename : str
        El nom del fitxer (ex: 'train_000_14.png').

    Returns
    -------
    int
        El número on ha caigut la bola.
    """
    # Eliminem l'extensió i separem per guions baixos
    name_parts = filename.replace('.png', '').split('_')
    
    # El número és l'última part de la llista
    return int(name_parts[2])

def prepare_dataset_structure(base_path: str):
    """
    Crea les 37 carpetes (del 0 al 36) per organitzar els patches.

    Parameters
    ----------
    base_path : str
        Ruta on es guardaran els patches classificats.
    """
    base = Path(base_path)
    for i in range(37):
        (base / str(i)).mkdir(parents=True, exist_ok=True)

def desar_patch(patch, ruta_desti, nom_original): #Farà falta esborrar totes les carpetes al final
    """
    Guarda el retall de la bola en la carpeta de la seva classe.
    
    Parameters
    ----------
    patch : np.ndarray
        La imatge 50x50 de la bola.
    ruta_desti : str
        Carpeta de la classe (ex: 'dataset_patches/14').
    nom_original : str
        Nom per conservar la traçabilitat.
    """
    cv2.imwrite(os.path.join(ruta_desti, f"patch_{nom_original}"), patch)