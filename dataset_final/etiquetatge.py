import os
import cv2

opcio = input("Quina carpeta vols etiquetar? (train/test): ").strip().lower()
current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, opcio)

if not os.path.exists(target_path):
    print(f"Error: No s'ha trobat la carpeta {opcio}")
    exit()

files = sorted([f for f in os.listdir(target_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for index, filename in enumerate(files):
    img_id = str(index).zfill(3)
    img_old_path = os.path.join(target_path, filename)
    
    img = cv2.imread(img_old_path)
    if img is None:
        continue
        
    display_img = cv2.resize(img, (800, 600)) 
    
    cv2.imshow("Etiquetador", display_img)
    cv2.waitKey(500)
    cv2.setWindowProperty("Etiquetador", cv2.WND_PROP_TOPMOST, 1)

    resultat = input(f"[{opcio}][{img_id}] Número: ")
    
    if resultat.lower() == 'q': break
    if resultat.lower() == 's': continue
    
    new_name = f"{opcio}_{img_id}_{resultat}.png"
    img_new_path = os.path.join(target_path, new_name)
    
    cv2.destroyAllWindows()
    
    try:
        os.rename(img_old_path, img_new_path)
        print(f"Fet: {new_name}")
    except Exception as e:
        print(f"Error: {e}")

cv2.destroyAllWindows()