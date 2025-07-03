from clusters import KMeans
import cv2
import csv
import os

def hsv_to_csv(nom_dossier, nom_nouveau_csv):

    def add_to_csv(nom_csv, dico):

        nouvelle_ligne = []

        for i in dico :
            h_norme = dico[i][0][0]/255
            s_norme = dico[i][0][1]/255
            v_norme = dico[i][0][2]/255

            nouvelle_ligne += [h_norme, s_norme, v_norme]

        with open(nom_csv, mode='a', newline='') as fichier_csv:
            writer = csv.writer(fichier_csv)
            writer.writerow(nouvelle_ligne)


    dataset_hsv_dir = nom_dossier
    image_list = [cv2.imread(os.path.join(dataset_hsv_dir, file)) for file in os.listdir(dataset_hsv_dir) if file.endswith(('.png','.PNG'))]

    with open(nom_nouveau_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['col1_h','col1_s','col1_v','col2_h','col2_s','col2_v','col3_h','col3_s','col3_v','col4_h','col4_s','col4_v'])


    for image in image_list :
        # Reshape the image to a 2D array of pixels
        dico = KMeans(n_clusters=4, demo=True, print_clusters=False).fit(image.reshape(-1, 3))
        add_to_csv(nom_nouveau_csv,dico)

hsv_to_csv("abstract_lab", "data_abstract2.csv")
