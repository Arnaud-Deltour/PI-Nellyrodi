import tkinter as tk
from tkinter import colorchooser
from tensorflow.keras.models import load_model
import numpy as np
import colorsys
import pandas as pd

# Mapping des styles à leurs fichiers modèles
model_paths = {
    "Art impressioniste": "impressionist_paintings.keras",
    "Art abstrait": "abstract_art.keras",
    "Mode et luxe": "fashion_luxury.keras",
}
models = {}

fenetre = tk.Tk()
fenetre.title("Sélecteur de couleurs artistiques")
fenetre.geometry("1000x700")
fenetre.configure(bg="#d9d9d9")

styles = list(model_paths.keys())
style_selectionne = tk.StringVar(value="")

# Polices et couleurs
font_title = ("Segoe UI", 20, "bold")
font_label = ("Segoe UI", 13)
font_small = ("Segoe UI", 10)
couleur_fond = "#d9d9d9"
couleur_fond_cadre = "#d9d9d9"
couleur_texte = "black"
couleur_accent = "#8ab4f8"
couleur_bouton = "#1a73e8"
couleur_bouton_hover = "#1967d2"

# Texte dynamique du style sélectionné
label_style_actuel = tk.Label(
    fenetre,
    text="Style sélectionné : Aucun",
    font=font_label,
    fg=couleur_texte,
    bg=couleur_fond,
)
label_style_actuel.pack(pady=(10, 5))


def texte_contraste(rgb):
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


def affichage(rgb, style):
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
    pred_hsv = np.array(pred[:9])

    couleurs = [
        rgb,
        colorsys.hsv_to_rgb(*pred_hsv[0:3]),
        colorsys.hsv_to_rgb(*pred_hsv[3:6]),
        colorsys.hsv_to_rgb(*pred_hsv[6:9]),
    ]

    couleurs = [tuple(int(c * 255) for c in couleur) for couleur in couleurs]
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
        return

    result = colorchooser.askcolor(title="Choisissez une couleur")
    rgb = result[0]

    if rgb:
        for widget in cadre_couleurs.winfo_children():
            widget.destroy()

        rgb_tuple = tuple(map(int, rgb))
        n, liste_couleurs = affichage(rgb_tuple, style_selectionne.get())

        # Couleur choisie
        cadre_choisie = tk.Frame(
            cadre_couleurs,
            bg=couleur_fond_cadre,
            bd=2,
            relief="groove",
            padx=10,
            pady=10,
        )
        cadre_choisie.pack(side="left", padx=10, pady=10)

        label = tk.Label(
            cadre_choisie,
            text="Couleur choisie",
            font=font_label,
            fg="white",
            bg=couleur_fond_cadre,
        )
        label.pack(pady=(0, 6))

        hex_color = "#%02x%02x%02x" % rgb_tuple
        case = tk.Frame(
            cadre_choisie,
            bg=hex_color,
            width=120,
            height=120,
            bd=3,
            relief="solid",
            highlightbackground=couleur_accent,
            highlightthickness=2,
        )
        case.pack()

        # Couleurs générées
        for i, couleur in enumerate(liste_couleurs[1:], start=1):
            cadre = tk.Frame(
                cadre_couleurs,
                bg=couleur_fond_cadre,
                bd=2,
                relief="groove",
                padx=10,
                pady=10,
            )
            cadre.pack(side="left", padx=10, pady=10)

            label = tk.Label(
                cadre,
                text=f"Couleur {i}",
                font=font_label,
                fg=couleur_texte,
                bg=couleur_fond_cadre,
            )
            label.pack(pady=(0, 6))

            hex_color = "#%02x%02x%02x" % couleur
            case = tk.Frame(
                cadre, bg=hex_color, width=110, height=110, bd=3, relief="ridge"
            )
            case.pack()

            label_rgb = tk.Label(
                case,
                text=str(couleur),
                font=font_small,
                fg=texte_contraste(couleur),
                bg=hex_color,
            )
            label_rgb.place(relx=0.5, rely=0.5, anchor="center")


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
    text="Sélectionnez un style artistique et découvrez des harmonies de couleurs inspirées",
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

# Cadre des couleurs
cadre_couleurs = tk.Frame(fenetre, bg=couleur_fond)
cadre_couleurs.pack(pady=15, fill="both", expand=True)

# Activation du bouton quand un style est choisi
style_selectionne.trace_add("write", activer_bouton)

fenetre.mainloop()
