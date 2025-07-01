import cv2
import os

# Chemin vers dossier images
#dataset_dir = 'PI-Nellyrodi\data\impressionist_paintings'
#dataset_hsv_dir = 'PI-Nellyrodi\data_hsv\impressionist_paintings'
dataset_dir = 'image'
dataset_hsv_dir = 'image'

file = 'IMG_4697.jpg'

# Liste des images dans le dossier
image_list = [os.path.join(dataset_dir, file)]
print(image_list)

i = 0
for image in image_list:
    img = cv2.imread(image)
    print(img)

    # Convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imwrite('image/img.jpg', hsv_img)
    #print(i)
    i+= 1

print("Conversion terminée")