import cv2
import numpy as np
from pathlib import Path
import preprocessing # El teu arxiu amb les funcions de resize, etc.

def crear_mascara_anell(dimensions, centre, radi_ext, radi_int):
    """Crea la màscara per aïllar la pista on roda la bola."""
    mascara = np.zeros(dimensions, dtype=np.uint8)
    cv2.circle(mascara, centre, radi_ext, 255, -1)
    cv2.circle(mascara, centre, radi_int, 0, -1)
    return mascara

def test_detectar_zero():
    # 1. Carregar rutes
    ruta_carpeta_train = Path(r'dataset_final/train')
    rutes_imatges = [arxiu for arxiu in ruta_carpeta_train.iterdir() if arxiu.suffix == '.png']
    
    DIMENSIONS = (800, 800)
    CENTRE = (400, 400)
    mascara_ruleta = crear_mascara_anell(DIMENSIONS, CENTRE, radi_ext=300, radi_int=220)
    
    # =======================================================
    # VALORS A PROVAR I AJUSTAR (Format HSV)
    # H (Color): 0-179 | S (Saturació): 0-255 | V (Brillantor): 0-255
    # =======================================================
    verd_baix = np.array([2, 3])  # Verd fosc/desaturat
    verd_alt = np.array([4]) # Verd clar/brillant
    
    # Provem només amb 10 imatges per anar ràpid
    for ruta in rutes_imatges[:10]: 
        imatge_original = cv2.imread(str(ruta))
        if imatge_original is None:
            continue
            
        # Preparar imatge
        imatge_redimensionada = preprocessing.resize(imatge_original, DIMENSIONS)
        
        # 2. Convertir a l'espai de color HSV
        hsv = cv2.cvtColor(imatge_redimensionada, cv2.COLOR_BGR2HSV)
        
        # 3. Aplicar els llindars de color per trobar el verd
        mascara_verda = cv2.inRange(hsv, verd_baix, verd_alt)
        
        # 4. Aplicar la màscara de l'anell (per ignorar el tapet de fora si també és verd)
        mascara_verda_final = cv2.bitwise_and(mascara_verda, mascara_verda, mask=mascara_ruleta)
        
        # 5. Visualització
        cv2.imshow('Imatge Original', imatge_redimensionada)
        cv2.imshow('Mascara Verda (Ajust)', mascara_verda_final)
        
        print(f"Mostrant {ruta.name}. Prem qualsevol tecla per veure la següent (o 'q' per sortir)...")
        tecla = cv2.waitKey(0)
        
        if tecla == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_detectar_zero()