import tkinter as tk
from tkinter import colorchooser
from tensorflow.keras.models import load_model
import numpy as np
import colorsys
import pandas as pd

# Chargement du modèle
model = load_model("impressionist_paintings.keras")

# --- Fenêtre principale ---
fenetre = tk.Tk()
fenetre.title("Sélecteur de couleur et affichage")
fenetre.geometry("1000x600")
fenetre.configure(bg="#2e2e2e")

# Liste des styles
styles = ["Art impressioniste", "Art abstrait", "Mode et luxe"]

# Variable pour le style (valeur initiale = vide)
style_selectionne = tk.StringVar(value="")

# Label dynamique pour afficher le style sélectionné
label_style_actuel = tk.Label(
    fenetre,
    text="Style sélectionné : Aucun",
    font=("Arial", 12),
    fg="white",
    bg="#2e2e2e",
)
label_style_actuel.pack()


def texte_contraste(rgb):
    # Calcule la luminosité perçue pour choisir la couleur du texte
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return "black" if luminance > 0.5 else "white"


def affichage(rgb, style):
    hsv = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    input = pd.DataFrame([[hsv[0], hsv[1], hsv[2]]]).values

    pred = model.predict(input)[0]
    pred_hsv = np.array([pred[i] for i in range(9)])

    couleurs = [
        (rgb[0], rgb[1], rgb[2]),
        colorsys.hsv_to_rgb(pred_hsv[0], pred_hsv[1], pred_hsv[2]),
        colorsys.hsv_to_rgb(pred_hsv[3], pred_hsv[4], pred_hsv[5]),
        colorsys.hsv_to_rgb(pred_hsv[6], pred_hsv[7], pred_hsv[8]),
    ]

    couleurs = [tuple(int(c * 255) for c in couleur) for couleur in couleurs]
    return len(couleurs), couleurs


def activer_bouton(*args):
    val = style_selectionne.get()
    if val in styles:
        bouton.config(state="normal")
        label_style_actuel.config(text=f"Style sélectionné : {val}")
        # Mise à jour de l'apparence du menu déroulant
        menu.config(
            bg="#2e2e2e",
            fg="white",
            activebackground="#454545",
            activeforeground="white",
        )
    else:
        bouton.config(state="disabled")
        label_style_actuel.config(text="Style sélectionné : Aucun")
        menu.config(bg="#2e2e2e", fg="white")


def choisir_couleur():
    if style_selectionne.get() not in styles:
        return

    result = colorchooser.askcolor(title="Choisissez une couleur")
    rgb = result[0]

    if rgb:
        for widget in cadre_couleurs.winfo_children():
            widget.destroy()

        rgb_tuple = tuple(map(int, rgb))
        n, liste_couleurs = affichage(rgb_tuple, style_selectionne.get())

        # Case spéciale "couleur choisie"
        hex_color = "#%02x%02x%02x" % rgb_tuple
        cadre_couleur_choisie = tk.Frame(cadre_couleurs)
        cadre_couleur_choisie.pack(side="left", padx=10, pady=10)

        label_choix_titre = tk.Label(
            cadre_couleur_choisie,
            text="Couleur choisie",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#2e2e2e",
        )
        label_choix_titre.pack()

        case_choisie = tk.Frame(
            cadre_couleur_choisie,
            bg=hex_color,
            width=110,
            height=110,
            bd=3,
            relief="solid",
        )
        case_choisie.pack(pady=5)
        label_choisie = tk.Label(
            case_choisie,
            text="",
            bg=hex_color,
            fg="white",
            font=("Arial", 10),
        )
        label_choisie.place(relx=0.5, rely=0.5, anchor="center")

        # Cases générées par le modèle avec titre et contraste du texte
        for i, couleur in enumerate(liste_couleurs[1:], start=1):
            hex_color = "#%02x%02x%02x" % couleur
            cadre_couleur = tk.Frame(cadre_couleurs)
            cadre_couleur.pack(side="left", padx=10, pady=10)

            label_titre_couleur = tk.Label(
                cadre_couleur,
                text=f"Couleur {i}",
                font=("Arial", 11, "bold"),
                fg="white",
                bg="#2e2e2e",
            )
            label_titre_couleur.pack()

            case = tk.Frame(
                cadre_couleur,
                bg=hex_color,
                width=90,
                height=90,
                bd=2,
                relief="raised",
            )
            case.pack(pady=5)
            label_rgb = tk.Label(
                case, text=str(couleur), bg=hex_color, fg=texte_contraste(couleur)
            )
            label_rgb.place(relx=0.5, rely=0.5, anchor="center")


# Titre
label_titre = tk.Label(
    fenetre,
    text="Générateur de palettes de couleurs harmonieuses en fonction de votre style",
    font=("Arial", 16),
    fg="white",
    bg="#2e2e2e",
)
label_titre.pack(pady=20)

# Menu déroulant pour les styles
label_menu = tk.Label(
    fenetre,
    text="Choisissez un style artistique :",
    font=("Arial", 12),
    fg="white",
    bg="#2e2e2e",
)
label_menu.pack()

menu = tk.OptionMenu(fenetre, style_selectionne, *styles)
menu.config(
    font=("Arial", 12),
    bg="#2e2e2e",
    fg="white",
    activebackground="#454545",
    activeforeground="white",
)
menu.pack(pady=5)

# Attacher la trace pour mise à jour dynamique
style_selectionne.trace_add("write", activer_bouton)

# Bouton pour choisir la couleur
bouton = tk.Button(
    fenetre,
    text="Choisir une couleur",
    font=("Arial", 14),
    command=choisir_couleur,
    state="disabled",  # Désactivé au départ car aucun style sélectionné
)
bouton.pack(pady=20)

# Cadre pour les couleurs affichées
cadre_couleurs = tk.Frame(fenetre, bg="#2e2e2e")
cadre_couleurs.pack(pady=20)

fenetre.mainloop()
