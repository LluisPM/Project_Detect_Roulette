import cv2
import matplotlib.pyplot as plt
from preprocessing import * # Importa les teves funcions


# 1. Carrega una imatge real del teu dataset
path = "/home/max/Codis/2n_Curs/SIVP/Projecte/Project_Detect_Roulette/dataset_final/train/train_000_14.png" 
img = cv2.imread(path)

if img is None:
    print(f"No s'ha trobat la imatge a: {path}")
else:
    # 2. Executem les funcions una a una per veure els passos
    img_gray = to_grayscale(img)
    img_blur = gaussian_blur(img_gray, kernel_size=7)
    img_aug  = data_augmentation(img) # Provem l'augmentation sobre l'original
    img_norm = normalize(img_gray) # La normalitzada per veure contrast

    # 3. Visualització amb Matplotlib
    titles = ['Original', 'Grises', 'Gaussian Blur', 'Data Augmentation', 'Normalized']
    images = [img, img_gray, img_blur, img_aug, img_norm]

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        
        # OpenCV usa BGR, Matplotlib usa RGB. Cal convertir per visualitzar bé:
        if len(images[i].shape) == 3:
            display_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        else:
            display_img = images[i] # Si és gris, no cal convertir
            
        plt.imshow(display_img, cmap='gray' if len(images[i].shape) == 2 else None)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()