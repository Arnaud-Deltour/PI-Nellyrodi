from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import cv2

#directions et noms des dossiers 
dossier_where_images_dir = "C:/Users/arnau/Downloads/images"
new_dataset_name ="abstract_compressed"

#ds = load_dataset("chashaotm/impressionist_paintings", split='train')
image_list = [cv2.imread(os.path.join(dossier_where_images_dir, file)) for file in os.listdir(dossier_where_images_dir) if file.endswith(('.png','.PNG'))]


from PIL import Image
import numpy as np
import os

os.makedirs(new_dataset_name, exist_ok=True)


def compress_image_without_new_colors(img: Image.Image, size=(100, 100)):
    #img = img.convert("RGB")
    np_img = np.array(img)

    h, w = np_img.shape[:2]
    target_h, target_w = size

    step_h = h // target_h
    step_w = w // target_w

    # Si l'image est trop petite, on peut la "centrer" dans un canevas blanc plus grand
    if step_h == 0 or step_w == 0:
        # Agrandir le canevas à la bonne taille avec fond blanc
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        np_img = np_img[:min(h, target_h), :min(w, target_w)]
        canvas[:np_img.shape[0], :np_img.shape[1]] = np_img
        return Image.fromarray(canvas)

    # Sélectionner un pixel dans chaque bloc sans interpolation
    compressed = np_img[::step_h, ::step_w]

    # Ajuster si la taille n'est pas pile 100x100
    compressed = compressed[:target_h, :target_w]

    return Image.fromarray(compressed)

# Boucle de test sur les premières images
for i in range(len(image_list)):
    img = image_list[i]
    compressed_img = compress_image_without_new_colors(img, size=(100, 100))

    # Sauvegarde
    compressed_img.save(f"{new_dataset_name}{i}.png")

'''
    # Affichage
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Compressed (100x100)")
    plt.imshow(compressed_img)
    plt.axis("off")

    plt.show()

'''
