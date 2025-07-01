import cv2
import os

# Chemin vers dossier images
dataset_dir = 'PI-Nellyrodi\data\impressionist_paintings'
dataset_hsv_dir = 'PI-Nellyrodi\data_hsv\impressionist_paintings'

# Liste des images dans le dossier
image_list = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith(('.jpg','.JPG'))]

i = 0
for image in image_list:
    img = cv2.imread(image)

    # Convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    name = image[29:]

    cv2.imwrite(os.path.join(dataset_hsv_dir,name), hsv_img)
    #print(i)
    i+= 1

print("Conversion terminée")