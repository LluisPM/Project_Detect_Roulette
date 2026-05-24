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
