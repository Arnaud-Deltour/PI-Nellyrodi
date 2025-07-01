from clusters import KMeans
import cv2
import csv
import os


def add_to_csv(nom_csv, dico):

    nouvelle_ligne = []

    for i in dico :
        h_norme = dico[i][0][0]/180
        s_norme = dico[i][0][1]/255
        v_norme = dico[i][0][2]/255

        nouvelle_ligne += [h_norme, s_norme, v_norme]

    with open(nom_csv, mode='a', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(nouvelle_ligne)


dataset_hsv_dir = 'data_hsv/impressionist_paintings'
image_list = [cv2.imread(os.path.join(dataset_hsv_dir, file)) for file in os.listdir(dataset_hsv_dir) if file.endswith(('.jpg','.JPG'))]


for image in image_list :
    dico = KMeans(n_clusters=4, max_iter=5).fit(image.reshape(-1,3))
    add_to_csv('data2.csv',dico)