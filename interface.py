import tkinter as tk
from tkinter import colorchooser
from tensorflow.keras.models import load_model
import numpy as np
import colorsys
import pandas as pd
import cv2

# Mapping des styles à leurs fichiers modèles
model_paths = {
    "Art impressioniste": "impressionist_paintings.keras",
    "Art abstrait": "abstract_art.keras",
    "Mode et luxe": "fashion_luxury.keras",
}
models = {}

fenetre = tk.Tk()
fenetre.title("Générateur de palettes de couleurs")
fenetre.geometry(f"{fenetre.winfo_screenwidth()}x{fenetre.winfo_screenheight()}")
fenetre.state("zoomed")  # Fonctionne sous Windows
#fenetre.attributes('-fullscreen', True)
fenetre.configure(bg="#ffffff")

styles = list(model_paths.keys())
style_selectionne = tk.StringVar(value="")

# Polices et couleurs
font_title = ("Segoe UI", 15, "bold")
font_label = ("Segoe UI", 10)
font_small = ("Segoe UI", 9)
couleur_fond = "#ffffff"
couleur_fond_cadre = "#d9d9d9"
couleur_texte = "black"
couleur_accent = "#8ab4f8"
couleur_bouton = "#1a73e8"
couleur_bouton_hover = "#1967d2"

# Texte dynamique des données sélectionnées
label_style_actuel = tk.Label(
    fenetre,
    text="Données sélectionnées : Aucune",
    font=font_label,
    fg=couleur_texte,
    bg=couleur_fond,
)
label_style_actuel.pack(pady=(10, 5))

#Détermine la couleur d'affichage du texte
def texte_contraste(rgb):
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"

# Récupère les couleurs issues de la prédiction par le perceptron
def get_colors(rgb, style):
    if style not in models:
        try:
            models[style] = load_model(model_paths[style])
        except Exception as e:
            print(f"Erreur : {e}")
            return 0, []

    model = models[style]

    hsv = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    input = pd.DataFrame([[hsv[0], hsv[1], hsv[2]]]).values

    pred = model.predict(input)[0]
    pred_lab = np.array(pred[:9])

    pal_converting = np.clip(pred_lab*255,0,255).astype(int)
    pal_converting = np.array([np.array([pal_converting[0:3],pal_converting[3:6],pal_converting[6:9]])])
    pal_converted = cv2.cvtColor(np.array(pal_converting, dtype=np.uint8), cv2.COLOR_LAB2RGB)

    # Couleurs à afficher
    couleurs = [
        rgb,
        pal_converted[0][0],
        pal_converted[0][1],
        pal_converted[0][2],
    ]

    #couleurs = liste de 4 tuples en RGB de 0 à 255
    couleurs = [tuple(int(c) for c in couleur) for couleur in couleurs]
    #print("Couleurs post tuple : ",couleurs)
    return len(couleurs), couleurs


def activer_bouton(*args):
    val = style_selectionne.get()
    if val in styles:
        bouton.config(state="normal")
        label_style_actuel.config(text=f"Style sélectionné : {val}")
    else:
        bouton.config(state="disabled")
        label_style_actuel.config(text="Style sélectionné : Aucun")

def choisir_couleur():
    if style_selectionne.get() not in styles:
        return None

    result = colorchooser.askcolor(title="Choisissez une couleur")
    input_rgb = result[0]

    if input_rgb:
        for widget in cadre_couleurs.winfo_children():
            widget.destroy()

        input_rgb_tuple = tuple(map(int, input_rgb))
        _, liste_couleurs = get_colors(input_rgb_tuple, style_selectionne.get())

        # Container for the palette
        palette_frame = tk.Frame(cadre_couleurs, bg=couleur_fond)
        palette_frame.pack(pady=0)

        for i, couleur in enumerate(liste_couleurs):
            hex_color = "#%02x%02x%02x" % couleur

            # Create a vertical container for the color + label
            color_container = tk.Frame(palette_frame, bg=couleur_fond_cadre)
            color_container.grid(row=0, column=i, padx=0, pady=0)

            # The color block
            case = tk.Frame(
                color_container,
                bg=hex_color,
                width=150,   # width of each block
                height=65,
                bd=0,
                relief="flat"
            )
            case.pack()

            # The RGB label below
            label_rgb = tk.Label(
                color_container,
                text=str(couleur),
                font=font_small,
                fg="black",
                bg=couleur_fond_cadre,
            )
            label_rgb.pack(pady=4)


# Titre principal
label_titre = tk.Label(
    fenetre,
    text="Générateur de palettes harmonieuses",
    font=font_title,
    fg="blue",
    bg=couleur_fond,
)
label_titre.pack(pady=10)

# Sous-titre
label_sous = tk.Label(
    fenetre,
    text="Sélectionnez une base de données et découvrez des harmonies de couleurs qui en sont inspirées",
    font=("Segoe UI", 12, "italic"),
    fg="black",
    bg=couleur_fond,
)
label_sous.pack(pady=(0, 15))

# Menu déroulant
cadre_menu = tk.Frame(fenetre, bg=couleur_fond)
cadre_menu.pack(pady=10)

label_menu = tk.Label(
    cadre_menu,
    text="Style artistique :",
    font=font_label,
    fg=couleur_texte,
    bg=couleur_fond,
)
label_menu.grid(row=0, column=0, sticky="w")

menu = tk.OptionMenu(cadre_menu, style_selectionne, *styles)
menu.config(
    font=font_label,
    bg=couleur_fond_cadre,
    fg=couleur_texte,
    activebackground="#444",
    activeforeground="white",
    bd=0,
    highlightthickness=0,
    relief="flat",
    width=20,
)
menu.grid(row=1, column=0, pady=6, ipadx=6, ipady=4)

# Bouton
bouton = tk.Button(
    fenetre,
    text="Choisir une couleur",
    font=font_label,
    command=choisir_couleur,
    state="disabled",
    bg=couleur_bouton,
    fg="white",
    activebackground=couleur_bouton_hover,
    activeforeground="white",
    bd=0,
    relief="flat",
    padx=12,
    pady=8,
    cursor="hand2",
)
bouton.pack(pady=15)


def on_enter(e):
    e.widget.config(bg=couleur_bouton_hover)


def on_leave(e):
    e.widget.config(bg=couleur_bouton)


bouton.bind("<Enter>", on_enter)
bouton.bind("<Leave>", on_leave)

# Zone des résultats avec fond gris
zone_resultats = tk.Frame(fenetre, bg=couleur_fond_cadre)
zone_resultats.pack(fill="both", expand=True)

# Cadre des couleurs à l'intérieur de la zone grise
cadre_couleurs = tk.Frame(zone_resultats, bg=couleur_fond_cadre)
cadre_couleurs.pack(pady=20)


# Activation du bouton quand un style est choisi
style_selectionne.trace_add("write", activer_bouton)

fenetre.mainloop()
