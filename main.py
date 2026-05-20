
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import ball_detector
import detectar_0

def processar_dataset(ruta_carpeta, mascara, dimensions):
    """
    Recorre totes les imatges d'una carpeta, extreu l'angle (X) i l'etiqueta (Y).
    """
    rutes_imatges = [arxiu for arxiu in ruta_carpeta.iterdir() if arxiu.suffix == '.png']
    
    X_data = []
    Y_labels = []
    
    for ruta in rutes_imatges:
        # Obtenim l'angle (X) i el número guanyador (Y)
        feature_x, etiqueta_y = detectar_0.extreure_dades_ml_relatives(ruta, mascara, dimensions)
        
        if feature_x is not None and etiqueta_y is not None:
            X_data.append(feature_x)
            Y_labels.append(etiqueta_y)
            
    return np.array(X_data), np.array(Y_labels)

def pipeline_A(ruta_train,ruta_test,mascara_ruleta,DIMENSIONS):
    print("Processant imatges d'entrenament...")
    X_train, Y_train = processar_dataset(ruta_train, mascara_ruleta, DIMENSIONS)
    print(f"S'han extret dades de {len(X_train)} imatges de train.")
    
    # 3. Extracció de dades (Test)
    print("Processant imatges de test...")
    X_test, Y_test = processar_dataset(ruta_test, mascara_ruleta, DIMENSIONS)
    print(f"S'han extret dades de {len(X_test)} imatges de test.")
    
    # 4. Entrenament del Model de Classificació
    print("Entrenant el model...")
    model = KNeighborsClassifier(n_neighbors=3) 
    model.fit(X_train, Y_train)
    
    # 5. Avaluació i Mètriques
    print("Avaluant el sistema...")
    Y_pred = model.predict(X_test)
    
    precisio = accuracy_score(Y_test, Y_pred)
    print(f"\n--- RESULTATS ---")
    print(f"Precisió del model (Accuracy): {precisio * 100:.2f}%")

def main():
    DIMENSIONS = (800, 800)
    CENTRE = (400, 400)
    mascara_ruleta = ball_detector.crear_mascara_anell(DIMENSIONS, CENTRE, radi_ext=300, radi_int=220)
    
    # Rutes de les dades (assegura't que les rutes existeixen)
    ruta_train = Path(r'dataset_final/train')
    ruta_test = Path(r'dataset_final/test')
    
    pipeline_A(ruta_train,ruta_test,mascara_ruleta,DIMENSIONS)

'''
if __name__ == "__main__":
    main()

'''

