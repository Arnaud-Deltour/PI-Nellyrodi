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
fenetre.geometry("1000x650")
fenetre.configure(bg="#1f1f1f")  # fond plus sombre pour contraste

# Liste des styles
styles = ["Art impressioniste", "Art abstrait", "Mode et luxe"]

# Variable pour le style (valeur initiale = vide)
style_selectionne = tk.StringVar(value="")

# Font globales
font_title = ("Segoe UI", 18, "bold")
font_label = ("Segoe UI", 13)
font_small = ("Segoe UI", 10)

# Label dynamique pour afficher le style sélectionné
label_style_actuel = tk.Label(
    fenetre,
    text="Style sélectionné : Aucun",
    font=font_label,
    fg="#eeeeee",
    bg="#1f1f1f",
)
label_style_actuel.pack(pady=(10, 5))


def texte_contraste(rgb):
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


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
        # Apparence menu après sélection
        menu.config(
            bg="#2e2e2e",
            fg="#f0f0f0",
            activebackground="#454545",
            activeforeground="#f0f0f0",
        )
    else:
        bouton.config(state="disabled")
        label_style_actuel.config(text="Style sélectionné : Aucun")
        menu.config(bg="#2e2e2e", fg="#f0f0f0")


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

        # Conteneur des couleurs (avec bordure et ombre légère)
        cadre_couleur_choisie = tk.Frame(
            cadre_couleurs, bg="#2e2e2e", bd=2, relief="ridge", padx=8, pady=8
        )
        cadre_couleur_choisie.pack(side="left", padx=12, pady=12)

        label_choix_titre = tk.Label(
            cadre_couleur_choisie,
            text="Couleur choisie",
            font=font_label,
            fg="#f0f0f0",
            bg="#2e2e2e",
        )
        label_choix_titre.pack(pady=(0, 6))

        hex_color = "#%02x%02x%02x" % rgb_tuple
        case_choisie = tk.Frame(
            cadre_couleur_choisie,
            bg=hex_color,
            width=120,
            height=120,
            bd=4,
            relief="groove",
            highlightthickness=2,
            highlightbackground="#ff9f1c",
        )
        case_choisie.pack()

        # Cases générées par le modèle
        for i, couleur in enumerate(liste_couleurs[1:], start=1):
            cadre_couleur = tk.Frame(
                cadre_couleurs, bg="#2e2e2e", bd=2, relief="ridge", padx=6, pady=6
            )
            cadre_couleur.pack(side="left", padx=12, pady=12)

            label_titre_couleur = tk.Label(
                cadre_couleur,
                text=f"Couleur {i}",
                font=font_label,
                fg="#f0f0f0",
                bg="#2e2e2e",
            )
            label_titre_couleur.pack(pady=(0, 6))

            hex_color = "#%02x%02x%02x" % couleur
            case = tk.Frame(
                cadre_couleur,
                bg=hex_color,
                width=110,
                height=110,
                bd=3,
                relief="raised",
            )
            case.pack()

            label_rgb = tk.Label(
                case,
                text=str(couleur),
                bg=hex_color,
                fg=texte_contraste(couleur),
                font=font_small,
            )
            label_rgb.place(relx=0.5, rely=0.5, anchor="center")


# Titre principal
label_titre = tk.Label(
    fenetre,
    text="Générateur de palettes de couleurs harmonieuses",
    font=font_title,
    fg="#f0f0f0",
    bg="#1f1f1f",
    pady=10,
)
label_titre.pack()

# Description / Sous-titre
label_sous_titre = tk.Label(
    fenetre,
    text="Choisissez un style artistique puis une couleur pour découvrir des harmonies adaptées",
    font=("Segoe UI", 12, "italic"),
    fg="#bbbbbb",
    bg="#1f1f1f",
    pady=5,
)
label_sous_titre.pack()

# Cadre du menu déroulant avec marge
cadre_menu = tk.Frame(fenetre, bg="#1f1f1f")
cadre_menu.pack(pady=15)

label_menu = tk.Label(
    cadre_menu,
    text="Choisissez un style artistique :",
    font=font_label,
    fg="#e0e0e0",
    bg="#1f1f1f",
)
label_menu.grid(row=0, column=0, sticky="w")

menu = tk.OptionMenu(cadre_menu, style_selectionne, *styles)
menu.config(
    font=font_label,
    bg="#2e2e2e",
    fg="#f0f0f0",
    activebackground="#454545",
    activeforeground="#f0f0f0",
    bd=0,
    highlightthickness=0,
)
menu.grid(row=1, column=0, sticky="w", pady=6, ipadx=10, ipady=4)

# Bouton choix couleur stylé avec hover
bouton = tk.Button(
    fenetre,
    text="Choisir une couleur",
    font=font_label,
    command=choisir_couleur,
    state="disabled",
    bg="#ff9f1c",
    fg="#1f1f1f",
    activebackground="#e68a00",
    activeforeground="#ffffff",
    bd=0,
    relief="ridge",
    padx=14,
    pady=8,
    cursor="hand2",
)
bouton.pack(pady=15)


# Ajout d'effet hover sur bouton
def on_enter(e):
    e.widget.config(bg="#e68a00")


def on_leave(e):
    e.widget.config(bg="#ff9f1c")


bouton.bind("<Enter>", on_enter)
bouton.bind("<Leave>", on_leave)

# Cadre pour les couleurs affichées
cadre_couleurs = tk.Frame(fenetre, bg="#1f1f1f")
cadre_couleurs.pack(pady=10, fill="both", expand=True)

# Trace pour gestion du bouton et menu
style_selectionne.trace_add("write", activer_bouton)

fenetre.mainloop()
