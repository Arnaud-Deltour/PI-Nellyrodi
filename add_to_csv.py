import csv


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