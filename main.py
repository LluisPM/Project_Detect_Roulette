import cv2
import preprocessing
import ball_detector
from pathlib import Path

# Configuració constant
CENTER = (400, 400)
R_IN, R_OUT = 220, 300

def run_detection_pipeline(image_path):
    # 1. Carregar i Preprocessar
    img = cv2.imread(str(image_path))
    img_res = preprocessing.resize(img, (800, 800))
    img_gray = preprocessing.to_grayscale(img_res)
    
    # 2. Localitzar Bola
    mask = ball_detector.get_annular_mask(img_gray.shape, CENTER, R_IN, R_OUT)
    coords = ball_detector.detect_ball_coords(img_gray, mask)
    
    if coords:
        # 3. Extreure Patch per a la IA
        patch = ball_detector.extract_patch(img_res, coords)
        
        # 4. Visualització (Opcional)
        cv2.circle(img_res, coords, 15, (0, 255, 0), 2)
        cv2.imshow("Deteccio", img_res)
        cv2.imshow("Patch IA", patch)
        cv2.waitKey(1)
        
        return patch
    return None

if __name__ == "__main__":
    # Bucle pel dataset
    ruta_images = Path('dataset_final/train').glob('*.png')
    for path in ruta_images:
        patch = run_detection_pipeline(path)
        