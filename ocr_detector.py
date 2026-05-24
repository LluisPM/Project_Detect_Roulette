import cv2
import numpy as np 
import ball_detector
from pathlib import Path
import easyocr

# Carreguem el lector un sol cop
lector = easyocr.Reader(['en'], gpu=False)

def alinear_ruleta(ruta_imatge, dimensions=(800, 800), centre=(400, 400)):
    imatge_original = cv2.imread(str(ruta_imatge))
    if imatge_original is None:
        return None, None, None
        
    img_color,img_suav = ball_detector.preparar_senyal(imatge_original, dimensions)
    mascara = ball_detector.crear_mascara_anell(dimensions, centre, 300, 220)
    img_emmascarada = ball_detector.aplicar_mascara(img_suav, mascara)
    
    cercles = ball_detector.extreure_caracteristiques_A(img_emmascarada)
    
    if cercles is None:
        return None, None, None
    
    bola_x = cercles[0][0][0]
    bola_y = cercles[0][0][1]
    centre_bola = (bola_x, bola_y)

    angle_graus = ball_detector.calcular_angle_bola(centre, centre_bola)
    angle_rotacio = angle_graus + 90

    matriu_rotacio = cv2.getRotationMatrix2D(centre, angle_rotacio, 1.0)
    img_rotada = cv2.warpAffine(img_color, matriu_rotacio, dimensions)

    return img_rotada, bola_x, bola_y 

def retallar(img_rotada, bola_x, bola_y, dimensions=(800,800), centre=(400,400)):
    dx = bola_x - centre[0]
    dy = bola_y - centre[1]
    radi_bola = np.sqrt(dx**2 + dy**2)
    
    offset = 60 
    y_numero = int(centre[1] - radi_bola - offset) 
    x_numero = int(centre[0])
    
    y_min = max(0, y_numero - 30)
    y_max = min(dimensions[1], y_numero + 30)
    x_min = max(0, x_numero - 40)
    x_max = min(dimensions[0], x_numero + 40)
    
    roi_numero = img_rotada[y_min:y_max, x_min:x_max]

    return roi_numero

def aplicar_ocr_easy(roi_numero):
    if roi_numero is None or roi_numero.size == 0:
        return ""
        
    img_neta = roi_numero.copy()
    h, w = img_neta.shape[:2]
    
    marge_lateral = 12
    img_neta[0:h, 0:marge_lateral] = [255, 255, 255]      
    img_neta[0:h, w-marge_lateral:w] = [255, 255, 255]    
    marge_superior = 6
    img_neta[0:marge_superior, 0:w] = [255, 255, 255]
    
    img_gran = cv2.resize(img_neta, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_gran, cv2.COLOR_BGR2RGB)
    
    resultats = lector.readtext(img_rgb, allowlist='0123456789')
    
    text_final = ""
    for deteccio in resultats:
        text = deteccio[1]
        confianca = deteccio[2]
        
        if confianca > 0.45:
            text_final += text
            
    return text_final.strip()


def processar_dataset_ocr(ruta_carpeta_test):
    """
    Aquesta funció és la equivalent a 'processar_dataset' del Pipeline A.
    Només llegeix les imatges, aplica OCR i retorna llistes amb la realitat i la predicció.
    """
    y_true = []
    y_pred = []
    
    if not ruta_carpeta_test.exists():
        print("Error: No s'ha trobat la carpeta.")
        return y_true, y_pred
        
    for arxiu in ruta_carpeta_test.iterdir():
        if arxiu.suffix.lower() == '.png':
            
            nom_sense_ext = arxiu.stem
            numero_real = nom_sense_ext.split('_')[-1]
            
            img_rotada, bola_x, bola_y = alinear_ruleta(arxiu)
            
            if img_rotada is not None:
                roi_numero = retallar(img_rotada, bola_x, bola_y)
                
                if roi_numero.size > 0:
                    numero_predit = aplicar_ocr_easy(roi_numero)
                    
                    if numero_predit == "":
                        numero_predit = "ND"
                        
                    y_true.append(numero_real)
                    y_pred.append(numero_predit)
                    
    return y_true, y_pred