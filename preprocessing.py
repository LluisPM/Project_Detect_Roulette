import cv2
import numpy as np
from typing import Tuple
import random

def resize(image: np.ndarray,  dims: Tuple) -> np.ndarray:
    """
    Ajusta les dimensions de la imatge a les indicades.

    Parameters
    ----------
    image : np.ndarray
        Imatge a modificar.
    dims : Tuple
        Tupla amb les dimensions objectiu.
    """
    return cv2.resize(image, dims)
    
def to_grayscale(image) -> np.ndarray:
    """
    Pasa de RGB a escala de grisos.

    Parameters
    ----------
    image : np.ndarray
        Imatge a modificar.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

def gaussian_blur(image: np.ndarray, kernel_size: int=5) -> np.ndarray:
    """
    Aplica una filtre Gaussià a la imatge.

    Parameters
    ----------
    image : np.ndarray
        Imatge a modificar.
    kernel_size : int
        Valor de les dimensions del kernel.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalitza els píxels per a una distribució estadística uniforme.

    Parameters
    ----------
    image : np.ndarray
        Imatge a modificar.
    """
    return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def random_brightening(image: np.ndarray) -> np.ndarray:
    """
    Modifica la brillantor de l'imatge de forma aleatoria.

    Parameters
    ----------
    image : np.ndarray
        Imatge a modificar.
    """
    value = random.randint(-30,30)
    return cv2.convertScaleAbs(image, alpha=1.0, beta=value)
    
def random_rotating(image : np.ndarray, max_angle: int = 180) -> np.ndarray:
    """
    Rota la imatge de manera aleatòria.

    Parameters
    ----------
    image : np.ndarray
        Imatge a modificar.
    max_angle: int
        Angle màxim que es pot rotar la imatge.
    """
    (h, w) = image.shape[:2]
    centre = (w // 2, h // 2)

    angle = random.uniform(-max_angle, max_angle)

    M = cv2.getRotationMatrix2D(centre, angle, 1.0)

    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def data_augmentation(image: np.ndarray) -> np.darray:
    """
    Aplica les 3 modificacions.

    Parameters
    ----------
    image : np.ndarray
        Imatge a modificar.
    """
    if image is None:
        raise ValueError("ERROR: Imatge no inexistent i no processable.")
    
    new_image = image.copy()

    rotated = random_rotating(new_image)
    brightened = random_brightening(rotated)
    out = gaussian_blur(brightened)

    return out