import preprocessing
import cv2
import numpy as np 
from pathlib import Path

# ==========================================
# 1. FUNCIONS DE PREPARACIÓ I MÀSCARES
# ==========================================

def crear_mascara_anell(dimensions, centre, radi_ext, radi_int):
    """
    Genera una màscara binària en forma d'anell per delimitar la zona de cerca.

    Parameters
    ----------
    dimensions : tuple
        Mida de la imatge (alçada, amplada).
    centre : tuple
        Coordenades (x, y) del centre de la ruleta.
    radi_ext : int
        Radi del cercle exterior de l'anell.
    radi_int : int
        Radi del cercle interior (forat) per ocultar el centre de la ruleta.

    Returns
    -------
    np.ndarray
        Màscara binària on la pista és blanca (255) i la resta negre (0).
    """
    mascara = np.zeros(dimensions, dtype=np.uint8)
    cv2.circle(mascara, centre, radi_ext, 255, -1)
    cv2.circle(mascara, centre, radi_int, 0, -1)
    return mascara

def aplicar_mascara(imatge, mascara):
    """
    Executa una operació de bitwise AND per aplicar la màscara a la imatge.

    Parameters
    ----------
    imatge : np.ndarray
        Imatge d'entrada (normalment en escala de grisos).
    mascara : np.ndarray
        Màscara binària del mateix tamany que la imatge.

    Returns
    -------
    np.ndarray
        Imatge on només es conserva la informació dins de la màscara.
    """
    return cv2.bitwise_and(imatge, imatge, mask=mascara)

# ==========================================
# 2. FUNCIONS DEL PIPELINE (ETAPES DE ML)
# ==========================================

def preparar_senyal(imatge_original, dimensions):
    """
    Fase de Preprocessament: Adequació de la imatge per a l'anàlisi.

    Parameters
    ----------
    imatge_original : np.ndarray
        Imatge crua del dataset en format BGR.
    dimensions : tuple
        Dimensions objectiu per al redimensionament.

    Returns
    -------
    tuple
        (imatge_color, imatge_suavitzada) per a visualització i càlcul respectivament.
    """
    imatge_redimensionada = preprocessing.resize(imatge_original, dimensions)
    imatge_gris = preprocessing.to_grayscale(imatge_redimensionada)
    imatge_suavitzada = preprocessing.gaussian_blur(imatge_gris, kernel_size=5)
    
    return imatge_redimensionada, imatge_suavitzada

def extreure_caracteristiques_A(imatge_processada):
    """
    Extracció de Característiques Geomètriques mitjançant la Transformada de Hough.

    Parameters
    ----------
    imatge_processada : np.ndarray
        Imatge binaritzada o suavitzada on es buscaran els cercles.

    Returns
    -------
    np.ndarray or None
        Llista de cercles detectats [x, y, radi] o None si no se'n troba cap.
    """
    cercles = cv2.HoughCircles(imatge_processada, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                               param1=50, param2=25, minRadius=10, maxRadius=25) 
    return cercles

def visualitzar_i_avaluar(imatge_color, cercles, nom_finestra='Resultat'):
    """
    Interfície de validació: Dibuixa els resultats i gestiona la interacció.

    Parameters
    ----------
    imatge_color : np.ndarray
        Imatge de fons on es dibuixaran els marcadors.
    cercles : np.ndarray
        Dades dels cercles detectats per Hough.
    nom_finestra : str
        Títol de la finestra de visualització.

    Returns
    -------
    int
        Codi ASCII de la tecla premuda per l'usuari.
    """
    if cercles is not None:
        cercles_arrodonits = np.uint16(np.around(cercles))
        for i in cercles_arrodonits[0, :]:
            # Dibuixem el perímetre verd i el centre vermell per a cada bola
            cv2.circle(imatge_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(imatge_color, (i[0], i[1]), 2, (0, 0, 255), 3)
            
    cv2.imshow(nom_finestra, imatge_color)
    return cv2.waitKey(0)

# ==========================================
# 3. CONTROLADOR PRINCIPAL PER IMATGE
# ==========================================

def processar_imatge(ruta, mascara_ruleta, dimensions):
    """
    Orquestra el flux complet de detecció per a una única mostra del dataset.

    Parameters
    ----------
    ruta : Path
        Ruta del fitxer de la imatge a processar.
    mascara_ruleta : np.ndarray
        Màscara precalculada per a la sessió.
    dimensions : tuple
        Dimensions de treball (ex: 800x800).

    Returns
    -------
    bool
        Retorna False si l'usuari demana interrompre l'execució ('q'), True altrament.
    """
    imatge_original = cv2.imread(str(ruta))
    if imatge_original is None:
        return True 
        
    img_color, img_suavitzada = preparar_senyal(imatge_original, dimensions)
    img_emmascarada = aplicar_mascara(img_suavitzada, mascara_ruleta)
    resultats = extreure_caracteristiques_A(img_emmascarada)
    
    tecla = visualitzar_i_avaluar(img_color, resultats, nom_finestra=f'Hough: {ruta.name}')
    
    return False if tecla == ord('q') else True

def calcular_angle_bola(centre_ruleta, centre_bola):
    """
    Calcula l'angle de la bola respecte al centre de la ruleta.
    Aquesta dada és invariant a la posició absoluta de la càmera.

    Parameters
    ----------
    centre_ruleta : tuple
        Coordenades (x, y) del centre de la ruleta (ex: 400, 400).
    centre_bola : tuple
        Coordenades (x, y) del centre de la bola detectada per Hough.

    Returns
    -------
    float
        Angle en graus dins del rang [0, 360).
    """
    cx, cy = centre_ruleta
    bx, by = centre_bola
    
    # Calculem l'arc tangent dels increments (dy, dx)
    # np.arctan2 gestiona els signes per donar l'angle correcte en els 4 quadrants
    angle_rad = np.arctan2(by - cy, bx - cx)
    angle_deg = np.degrees(angle_rad)
    
    # Normalitzem perquè no hi hagi angles negatius
    if angle_deg < 0:
        angle_deg += 360
        
    return angle_deg

def extreure_patch_bola(imatge_color, centre_bola, mida_patch=50):
    """
    Extreu una sub-imatge (ROI) quadrada centrada en la posició de la bola.

    Parameters
    ----------
    imatge_color : np.ndarray
        La imatge original redimensionada a color (BGR).
    centre_bola : tuple
        Coordenades (x, y) on Hough ha detectat la bola.
    mida_patch : int
        Mida del costat del quadrat (per defecte 50x50 píxels).

    Returns
    -------
    np.ndarray
        El retall de la imatge llest per a ser classificat o guardat.
    """
    x, y = int(centre_bola[0]), int(centre_bola[1])
    radi_patch = mida_patch // 2
    
    # Definim els límits del retall (cropping)
    # Fem servir max/min per assegurar-nos que no sortim de la imatge
    y_min, y_max = max(0, y - radi_patch), y + radi_patch
    x_min, x_max = max(0, x - radi_patch), x + radi_patch
    
    patch = imatge_color[y_min:y_max, x_min:x_max]
    
    return patch

# ==========================================
# 4. EXECUCIÓ DEL PROGRAMA
# ==========================================

def main():
    """
    Punt d'entrada del script. Gestiona la càrrega del dataset i el bucle principal.
    """
    ruta_carpeta_train = Path(r'dataset_final/train')
    rutes_imatges = [arxiu for arxiu in ruta_carpeta_train.iterdir() if arxiu.suffix == '.png'] 
    
    DIMENSIONS = (800, 800)
    CENTRE = (400, 400)
    mascara_ruleta = crear_mascara_anell(DIMENSIONS, CENTRE, radi_ext=300, radi_int=220)
    
    for contador, ruta in enumerate(rutes_imatges):
        if contador >= 90:
            break
            
        if not processar_imatge(ruta, mascara_ruleta, DIMENSIONS):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()