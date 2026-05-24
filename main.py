import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import ball_detector
import detectar_0
import ocr_detector

DIMENSIONS = (800, 800)
CENTRE = (400, 400)
RUTA_TRAIN = Path('dataset_final/train')
RUTA_TEST = Path('dataset_final/test')


def processar_dataset(ruta_carpeta, mascara, dimensions):
    """
    Recorre totes les imatges d'una carpeta, extreu l'angle (X) i l'etiqueta (Y).
    """
    rutes_imatges = [arxiu for arxiu in ruta_carpeta.iterdir() if arxiu.suffix == '.png']
    
    X_data = []
    Y_labels = []
    
    for ruta in rutes_imatges:
        feature_x, etiqueta_y = detectar_0.extreure_dades_ml_relatives(ruta, mascara, dimensions)
        
        if feature_x is not None and etiqueta_y is not None:
            X_data.append(feature_x)
            Y_labels.append(etiqueta_y)
            
    return np.array(X_data), np.array(Y_labels)

def pipeline_A(ruta_train, ruta_test, mascara_ruleta, dimensions):
    print("\n--- PIPELINE A: KNN ---")
    print("Processant imatges d'entrenament...")
    X_train, Y_train = processar_dataset(ruta_train, mascara_ruleta, dimensions)
    print(f"S'han extret dades de {len(X_train)} imatges de train.")
    
    print("Processant imatges de test...")
    X_test, Y_test = processar_dataset(ruta_test, mascara_ruleta, dimensions)
    print(f"S'han extret dades de {len(X_test)} imatges de test.")
    
    print("Entrenant el model...")
    model = KNeighborsClassifier(n_neighbors=3) 
    model.fit(X_train, Y_train)
    
    print("Avaluant el sistema...")
    Y_pred = model.predict(X_test)
    
    precisio = accuracy_score(Y_test, Y_pred)
    print(f"\n--- RESULTATS ---")
    print(f"Precisió del model (Accuracy): {precisio * 100:.2f}%")

def pipeline_B(ruta_test):
    """
    Executa el pipeline complet de l'OCR i n'avalua el rendiment.
    Això és l'equivalent directe a 'pipeline_A'.
    """
    print("\n--- PIPELINE B: OCR ---")
    print("Processant imatges i avaluant OCR...")
    y_true, y_pred = ocr_detector.processar_dataset_ocr(ruta_test)
    
    total_imatges = len(y_true)
    if total_imatges == 0:
        print("No s'han pogut avaluar imatges.")
        return

    encerts = sum(1 for real, predit in zip(y_true, y_pred) if real == predit)
    precisio = encerts / total_imatges
    
    print(f"\n--- RESULTATS ---")
    print(f"Imatges avaluades: {total_imatges}")
    print(f"Encerts totals: {encerts}")
    print(f"Precisió del model (Accuracy): {precisio * 100:.2f}%\n")
    
    print("Generant la Matriu de Confusió...")
    etiquetes = [str(i) for i in range(37)] + ["ND"]
    cm = confusion_matrix(y_true, y_pred, labels=etiquetes)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetes)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
    
    plt.title('Matriu de Confusió - Reconeixement OCR de la Ruleta', fontsize=16)
    plt.xlabel('El que ha predit l\'OCR', fontsize=12)
    plt.ylabel('El número Real (Etiqueta)', fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    mascara_ruleta = ball_detector.crear_mascara_anell(DIMENSIONS, CENTRE, radi_ext=300, radi_int=220)
    
    # Demanem a l'usuari què vol fer i ho guardem a la variable 'opcio'
    opcio = input("Digues quin pipeline vols executar (A / B / TOTS): ").strip().upper()
    
    if opcio == 'A':
        pipeline_A(RUTA_TRAIN, RUTA_TEST, mascara_ruleta, DIMENSIONS)
    elif opcio == 'B':
        pipeline_B(RUTA_TEST)
    elif opcio == 'TOTS':
        pipeline_A(RUTA_TRAIN, RUTA_TEST, mascara_ruleta, DIMENSIONS)
        pipeline_B(RUTA_TEST)
    else:
        print("Opció no vàlida. Si us plau escriu A, B o TOTS.")

if __name__ == "__main__":
    main()