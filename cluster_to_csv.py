from add_to_csv import add_to_csv
from clusters import KMeans
import cv2
import glob

chemin_dossier = 'data_hsv/*.jpg'
chemins_images = glob.glob(chemin_dossier)
images = [cv2.imread(chemin) for chemin in chemins_images]


for photo in images :
    dico = KMeans(n_clusters=5).fit(photo.reshape(-1,3))
    add_to_csv('data.csv',dico)


chemin_dossier = 'data_hsv/*.jpg'
chemins_images = glob.glob(chemin_dossier)
