import tkinter as tk
from tkinter import colorchooser
import numpy as np
import pandas as pd
import cv2

# --- Chargement des données palettes ---
df = pd.read_csv("data_abstract2.csv")
df = df * 255  # Passage à l'échelle 0-255
n_palettes = len(df)
palettes = df.values.reshape((n_palettes, 4, 3))  # [n, 4, 3] en LAB
all_colors = palettes.reshape((-1, 3))  # toutes les couleurs

def lab_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def find_palettes_from_color(input_lab_color, R, n):
    palettes_proches = []
    for i, color in enumerate(all_colors):
        dist = lab_distance(color, input_lab_color)
        if dist <= R:
            idx_palette = i // 4
            palettes_proches.append((idx_palette, dist))
    
    # On élimine les doublons en gardant la distance minimale pour chaque palette
    dist_min_par_palette = {}
    for idx, dist in palettes_proches:
        if idx not in dist_min_par_palette or dist < dist_min_par_palette[idx]:
            dist_min_par_palette[idx] = dist
    
    # On trie par distance croissante
    palettes_tries = sorted(dist_min_par_palette.items(), key=lambda x: x[1])
    
    # On récupère les palettes triées, limité à n palettes
    palettes_selectionnees = [tuple(map(tuple, palettes[idx])) for idx, _ in palettes_tries[:n]]
    
    if len(palettes_selectionnees) == 0:
        return None
    else:
        return [np.array(p) for p in palettes_selectionnees]


# --- GUI ---

fenetre = tk.Tk()
fenetre.title("Sélecteur de couleurs artistiques")
fenetre.geometry("1000x700")
fenetre.configure(bg="#d9d9d9")

font_label = ("Segoe UI", 13)
font_small = ("Segoe UI", 10)
couleur_fond = "#d9d9d9"
couleur_fond_cadre = "#d9d9d9"
couleur_texte = "black"
couleur_accent = "#8ab4f8"
couleur_bouton = "#1a73e8"
couleur_bouton_hover = "#1967d2"

def texte_contraste(rgb):
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"

def affichage(rgb):
    lab = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0][0]
    R = 15  # rayon tolérance
    
    liste_pred = find_palettes_from_color(lab, R, 6)
    
    if not liste_pred:
        return 0, []
    
    toutes_palettes_rgb = []
    for pred in liste_pred:
        # Convertir palette LAB [4,3] en RGB [4,3]
        pal_converting = np.clip(pred, 0, 255).astype(np.uint8)
        pal_converting = np.array([pal_converting])
        pal_converted = cv2.cvtColor(pal_converting, cv2.COLOR_LAB2RGB)[0]
        
        # convertir en tuples RGB entiers
        palette_rgb = [tuple(int(c) for c in couleur) for couleur in pal_converted]
        toutes_palettes_rgb.append(palette_rgb)

    # Liste finale de palettes RGB (liste de listes de tuples RGB)
    return len(toutes_palettes_rgb), toutes_palettes_rgb

def choisir_couleur():
    result = colorchooser.askcolor(title="Choisissez une couleur")
    rgb = result[0]

    if rgb:
        for widget in cadre_couleurs.winfo_children():
            widget.destroy()

        rgb_tuple = tuple(map(int, rgb))
        n, liste_palettes = affichage(rgb_tuple)

        if n == 0:
            label_aucune = tk.Label(cadre_couleurs, text="Aucune palette trouvée", font=font_label)
            label_aucune.pack()
            return

        # Affichage couleur choisie
        cadre_choisie = tk.Frame(cadre_couleurs, bg=couleur_fond_cadre, bd=2, relief="groove", padx=10, pady=10)
        cadre_choisie.pack(side="left", padx=10, pady=10)

        label = tk.Label(cadre_choisie, text="Couleur choisie", font=font_label, fg="white", bg=couleur_fond_cadre)
        label.pack(pady=(0, 6))

        hex_color = "#%02x%02x%02x" % rgb_tuple
        case = tk.Frame(cadre_choisie, bg=hex_color, width=120, height=120, bd=3, relief="solid", highlightbackground=couleur_accent, highlightthickness=2)
        case.pack()

        # Affichage des palettes trouvées
        for i, palette in enumerate(liste_palettes, start=1):
            cadre = tk.Frame(cadre_couleurs, bg=couleur_fond_cadre, bd=2, relief="groove", padx=10, pady=10)
            cadre.pack(side="left", padx=10, pady=10)

            label = tk.Label(cadre, text=f"Palette {i}", font=font_label, fg=couleur_texte, bg=couleur_fond_cadre)
            label.pack(pady=(0, 6))

            for couleur in palette:
                hex_color = "#%02x%02x%02x" % couleur
                case = tk.Frame(cadre, bg=hex_color, width=70, height=70, bd=3, relief="ridge")
                case.pack(pady=3)

                label_rgb = tk.Label(case, text=str(couleur), font=font_small, fg=texte_contraste(couleur), bg=hex_color)
                label_rgb.place(relx=0.5, rely=0.5, anchor="center")


# Titre principal
label_titre = tk.Label(fenetre, text="Générateur de palettes harmonieuses", font=("Segoe UI", 20, "bold"), fg="blue", bg=couleur_fond)
label_titre.pack(pady=10)

# Bouton choisir couleur
bouton = tk.Button(fenetre, text="Choisir une couleur", font=font_label, command=choisir_couleur, bg=couleur_bouton, fg="white", activebackground=couleur_bouton_hover, activeforeground="white", bd=0, relief="flat", padx=12, pady=8, cursor="hand2")
bouton.pack(pady=15)

def on_enter(e):
    e.widget.config(bg=couleur_bouton_hover)
def on_leave(e):
    e.widget.config(bg=couleur_bouton)

bouton.bind("<Enter>", on_enter)
bouton.bind("<Leave>", on_leave)

# Cadre pour afficher les couleurs
cadre_couleurs = tk.Frame(fenetre, bg=couleur_fond)
cadre_couleurs.pack(pady=15, fill="both", expand=True)

fenetre.mainloop()
