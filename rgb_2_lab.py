import cv2
import os

# dataset_dir = 'abstract_compressed'
# dataset_hsv_dir = 'abstract_lab'
dataset_dir = "data/impressionist_compressed"
dataset_hsv_dir = "data/impressionist_lab"

# Créer dossier de sortie s'il n'existe pas
os.makedirs(dataset_hsv_dir, exist_ok=True)

# Liste des images dans le dossier
image_list = [
    os.path.join(dataset_dir, file)
    for file in os.listdir(dataset_dir)
    if file.lower().endswith(".png")
]

i = 0
for image_path in image_list:
    img = cv2.imread(image_path)

    # Convertir en LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Extraire nom de fichier proprement
    name = os.path.basename(image_path)

    # Sauvegarder l'image LAB
    cv2.imwrite(os.path.join(dataset_hsv_dir, name), lab_img)
    i += 1

print(f"Conversion terminée pour {i} images")
