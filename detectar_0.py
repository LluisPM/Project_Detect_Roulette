import preprocessing
import cv2
import numpy as np 
from pathlib import Path
import math
import ball_detector
from skimage import measure

def trobar_casella_zero(imatge_color, mascara_ruleta):
    """
    Segmentació per color per trobar el centroide de la casella del 0 (verda).
    """
    # 1. Convertim a l'espai de color HSV per separar bé el to del color
    hsv = cv2.cvtColor(imatge_color, cv2.COLOR_BGR2HSV)
    
    # Definim el rang del color verd en HSV 
    # (Aquests valors poden requerir petits ajustos empírics amb les teves fotos de 'train')
    verd_baix = np.array([6, 7])
    verd_alt = np.array([8])
    
    # Creem una màscara binària on només el verd sigui blanc
    mascara_verda = cv2.inRange(hsv, verd_baix, verd_alt)
    
    # Apliquem la màscara de l'anell perquè no busqui res verd fora de la ruleta
    mascara_verda = cv2.bitwise_and(mascara_verda, mascara_verda, mask=mascara_ruleta)
    
    cv2.imshow("Mascara Verda", mascara_verda)
    cv2.waitKey(0)
    # 2. Etiquetem les regions connectades i n'extraiem les propietats
    labels = measure.label(mascara_verda)
    props = measure.regionprops(labels)
    
    if not props:
        return None # No s'ha trobat color verd
        
    regio_zero = max(props, key=lambda p: p.area)
    
    y_zero, x_zero = regio_zero.centroid
    return (x_zero, y_zero)

def extreure_dades_ml_relatives(ruta, mascara_ruleta, dimensions):
    """
    Processa la imatge, troba la bola i el zero, i retorna l'angle RELATIU.
    """
    nom_fitxer = ruta.stem 
    etiqueta_y = int(nom_fitxer.split('_')[-1]) 
    
    imatge_original = cv2.imread(str(ruta))
    if imatge_original is None:
        return None, None
        
    img_color, img_suavitzada = ball_detector.preparar_senyal(imatge_original, dimensions)
    img_emmascarada = ball_detector.aplicar_mascara(img_suavitzada, mascara_ruleta)
    
    # A) Trobar la Bola
    cercles = ball_detector.extreure_caracteristiques_A(img_emmascarada)
    
    # B) Trobar el Zero
    centroide_zero = trobar_casella_zero(img_color, mascara_ruleta)
    
    if cercles is not None and centroide_zero is not None:
        cercles_arrodonits = np.uint16(np.around(cercles))
        x_bola = cercles_arrodonits
        y_bola = cercles_arrodonits[9]
        
        x_zero, y_zero = centroide_zero
        centre_x, centre_y = 400, 400
        
        angle_bola_rad = math.atan2(y_bola - centre_y, x_bola - centre_x)
        angle_zero_rad = math.atan2(y_zero - centre_y, x_zero - centre_x)
        
        angle_bola_graus = math.degrees(angle_bola_rad) % 360
        angle_zero_graus = math.degrees(angle_zero_rad) % 360
        
        # D) LA NOVA FEATURE: L'Angle Relatiu
        angle_relatiu = (angle_bola_graus - angle_zero_graus) % 360
        
        feature_x = [angle_relatiu] 
        return feature_x, etiqueta_y
        
    return None, None 

