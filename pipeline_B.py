import cv2
import numpy as np 
import ball_detector
from pathlib import Path
import easyocr
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    # Retornem també la posició de la bola per al següent pas
    return img_rotada, bola_x, bola_y 

def retallar(img_rotada, bola_x, bola_y, dimensions=(800,800), centre=(400,400)):
    dx = bola_x - centre[0]
    dy = bola_y - centre[1]
    radi_bola = np.sqrt(dx**2 + dy**2)
    
    # RESTÈM l'offset per anar cap a dalt (allunyar-nos del centre)
    # Segurament necessitaràs un valor entre 50 i 70. Prova amb 60 per començar.
    offset = 60 
    y_numero = int(centre[1] - radi_bola - offset) 
    x_numero = int(centre[0])
    
    # Fem el retall una mica més ample per a números de dues xifres
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
    
    # Deixem uns marges més suaus per no tallar el 14, 
    # assumint que pot entrar una mica de brossa
    marge_lateral = 12
    img_neta[0:h, 0:marge_lateral] = [255, 255, 255]      
    img_neta[0:h, w-marge_lateral:w] = [255, 255, 255]    
    marge_superior = 6
    img_neta[0:marge_superior, 0:w] = [255, 255, 255]
    
    img_gran = cv2.resize(img_neta, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_gran, cv2.COLOR_BGR2RGB)
    
    cv2.imshow('Imatge enviada al OCR', img_gran)
    
    # Llegim el text. Ara ens retornarà llistes amb: [Coordenades, Text, Confiança]
    resultats = lector.readtext(img_rgb, allowlist='0123456789')
    
    text_final = ""
    
    for deteccio in resultats:
        coordenades = deteccio[0]
        text = deteccio[1]
        confianca = deteccio[2]
        
        
        # Filtrem: Només ens quedem amb el text si el lector està molt segur (més del 50%)
        # El soroll de la sanefa acostuma a donar confiances molt baixes (10% - 30%)
        if confianca > 0.45:
            text_final += text
            
    return text_final.strip()

def prova2():
    ruta_carpeta_test = Path('dataset_final/test') 
    if not ruta_carpeta_test.exists():
        print("Error: No s'ha trobat la carpeta.")
        return
        
    print("Processant imatges i avaluant OCR...")
    
    total_imatges = 0
    encerts = 0
    
    # Llistes per alimentar la Matriu de Confusió
    y_true = []
    y_pred = []
    
    for arxiu in ruta_carpeta_test.iterdir():
        if arxiu.suffix.lower() == '.png':
            
            nom_sense_ext = arxiu.stem
            numero_real = nom_sense_ext.split('_')[-1]
            
            img_rotada, bola_x, bola_y = alinear_ruleta(arxiu)
            
            if img_rotada is not None:
                roi_numero = retallar(img_rotada, bola_x, bola_y)
                
                if roi_numero.size > 0:
                    # Com que ja no tens l'imshow dins d'aquesta funció, 
                    # s'executarà tota la carpeta molt ràpid.
                    numero_predit = aplicar_ocr_easy(roi_numero)
                    
                    # Si l'OCR es queda en blanc, ho marquem com a "ND" (No Detectat)
                    if numero_predit == "":
                        numero_predit = "ND"
                        
                    # Guardem la realitat i la predicció
                    y_true.append(numero_real)
                    y_pred.append(numero_predit)
                    
                    if numero_predit == numero_real:
                        encerts += 1
                        
                    total_imatges += 1
            
    if total_imatges > 0:
        precisio = encerts / total_imatges
        print(f"\n--- RESULTATS ---")
        print(f"Imatges avaluades: {total_imatges}")
        print(f"Encerts totals: {encerts}")
        print(f"Precisió del model (Accuracy): {precisio * 100:.2f}%\n")
        
        # ==========================================
        # GENERACIÓ DE LA MATRIU DE CONFUSIÓ
        # ==========================================
        print("Generant la Matriu de Confusió...")
        
        # Definim totes les caselles possibles de la ruleta (del '0' al '36') + 'ND' pels errors en blanc
        etiquetes = [str(i) for i in range(37)] + ["ND"]
        
        # Calculem la matriu matemàtica
        cm = confusion_matrix(y_true, y_pred, labels=etiquetes)
        
        # Preparem la figura gran perquè hi caben 38 columnes i no s'aixafin els números
        fig, ax = plt.subplots(figsize=(16, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetes)
        
        # Dibuixem amb un mapa de colors blau
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
        
        plt.title('Matriu de Confusió - Reconeixement OCR de la Ruleta', fontsize=16)
        plt.xlabel('El que ha predit l\'OCR', fontsize=12)
        plt.ylabel('El número Real (Etiqueta)', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("No s'han pogut avaluar imatges.")

if __name__ == '__main__':
    prova2()