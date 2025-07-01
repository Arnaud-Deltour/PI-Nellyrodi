import tkinter as tk
from tkinter import colorchooser
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset
import numpy as np
import colorsys
import pandas as pd

model = load_model('impressionist_paintings.keras')

def affichage(rgb):
    """
    Simule une fonction qui prend une couleur RGB et renvoie :
    - un entier n (nombre de cases)
    - une liste de n couleurs RGB (variations simulées ici)
    """
    hsv = colorsys.rgb_to_hsv(rgb[0]/255,rgb[1]/255,rgb[2]/255)
    input = pd.DataFrame([[hsv[0], hsv[1], hsv[2]]]).values
    couleurs = [colorsys.hsv_to_rgb(hsv[0],hsv[1],hsv[2])]
    print(couleurs)
    n = 1

    pred = model.predict(input)[0]
    pred_hsv = np.array([pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7], pred[8]])
    print(pred_hsv)
    couleurs = [colorsys.hsv_to_rgb(hsv[0],hsv[1],hsv[2]), colorsys.hsv_to_rgb(pred_hsv[0],pred_hsv[1],pred_hsv[2]), colorsys.hsv_to_rgb(pred_hsv[3],pred_hsv[4],pred_hsv[5]), colorsys.hsv_to_rgb(pred_hsv[6],pred_hsv[7],pred_hsv[8])]
    couleurs = [tuple(int(couleurs[i][j]*255) for j in range(len(couleurs[i]))) for i in range(len(couleurs))]
    print(couleurs)

    return n, couleurs


def choisir_couleur():
    result = colorchooser.askcolor(title="Choisissez une couleur")
    rgb = result[0]  # (R, G, B)
    if rgb:
        # Nettoyer l'ancien affichage
        for widget in cadre_couleurs.winfo_children():
            widget.destroy()

        n, liste_couleurs = affichage(tuple(map(int, rgb)))

        # Créer n cases avec les couleurs
        for couleur in liste_couleurs:
            hex_color = "#%02x%02x%02x" % couleur
            case = tk.Frame(
                cadre_couleurs,
                bg=hex_color,
                width=100,
                height=100,
                bd=2,
                relief="raised",
            )
            case.pack(side="left", padx=10, pady=10)
            # Affiche le code RGB dans la case
            label_rgb = tk.Label(case, text=str(couleur), bg=hex_color, fg="white")
            label_rgb.place(relx=0.5, rely=0.5, anchor="center")


# Fenêtre principale
fenetre = tk.Tk()
fenetre.title("Sélecteur de couleur et affichage")
fenetre.geometry("1000x600")

label = tk.Label(
    fenetre,
    text="Générateur d'une palette de quelques couleurs qui vont bien avec la vôtre",
    font=("Arial", 16),
)
label.pack(pady=50)

# Bouton pour choisir la couleur
bouton = tk.Button(
    fenetre, text="Choisir une couleur", font=("Arial", 14), command=choisir_couleur
)
bouton.pack(pady=20)

# Cadre pour afficher les cases colorées
cadre_couleurs = tk.Frame(fenetre)
cadre_couleurs.pack(pady=20)

fenetre.mainloop()
