import preprocessing
import cv2
import os
import numpy as np 
from pathlib import Path

def pipeline_A(imatge_processada):
    cercles = cv2.HoughCircles(imatge_processada, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                               param1=50, param2=25, 
                               minRadius=10, 
                               maxRadius=25) 
    return cercles
if __name__== '__main__':
    #Carreguem el dataset
    ruta_carpeta_train = Path(r'dataset_final/train')
    rutes_imatges = [arxiu for arxiu in ruta_carpeta_train.iterdir() if arxiu.suffix == '.png'] 
    contador = 0
    for ruta in rutes_imatges:
        contador +=1
        imatge_original = cv2.imread(str(ruta))
        imatge_redimensionada = preprocessing.resize(imatge_original,800)
        imatge_gris = preprocessing.to_grayscale(imatge_redimensionada)
        imatge_suavitzada = preprocessing.gaussian_blur(imatge_gris,kernel_size=5)
        mascara = np.zeros((800, 800), dtype=np.uint8)
        
        centre_x, centre_y = 400, 400 # El centre exacte de la imatge de 800x800
        
        # Dibuixem un cercle blanc ple: Aquest és el límit EXTERIOR on pot rodar la bola
        radi_exterior = 300 # <-- HAURÀS D'AJUSTAR AQUEST NÚMERO
        cv2.circle(mascara, (centre_x, centre_y), radi_exterior, 255, -1)
        
        # Dibuixem un cercle negre ple: Aquest és el límit INTERIOR (tapem el centre de la ruleta)
        radi_interior = 220 # <-- HAURÀS D'AJUSTAR AQUEST NÚMERO
        cv2.circle(mascara, (centre_x, centre_y), radi_interior, 0, -1)
        
        # Apliquem la màscara a la imatge suavitzada
        imatge_emmascarada = cv2.bitwise_and(imatge_suavitzada, imatge_suavitzada, mask=mascara)
        resultat_hough = pipeline_A(imatge_emmascarada)
        if contador < 90:
            if resultat_hough is not None:
                cercles_arrodonits = np.uint16(np.around(resultat_hough))
            
                for i in cercles_arrodonits[0, :]:

                    cv2.circle(imatge_redimensionada, (i[0], i[1]), i[2], (0, 255, 0), 2)

                    cv2.circle(imatge_redimensionada, (i[0], i[1]), 2, (0, 0, 255), 3)

            cv2.imshow('Hough Cercles', imatge_redimensionada)
                
            cv2.waitKey(0) 
        else:
            break