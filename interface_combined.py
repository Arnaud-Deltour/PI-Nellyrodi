import tkinter as tk
from tkinter import colorchooser
from tensorflow.keras.models import load_model
import numpy as np
import colorsys
import pandas as pd
import cv2

color_w = 700

# Mapping des styles à leurs fichiers modèles
model_paths = {
    "Art impressioniste": ["impressionist_paintings1.keras","impressionist_paintings2.keras","impressionist_paintings3.keras"],
    "Art abstrait": ["abstract_art1.keras","abstract_art2.keras","abstract_art3.keras"]}
models = {"Art impressioniste":"","Art abstrait":""}
csv_paths = {"Art impressioniste":"data_impressionist.csv","Art abstrait":"data_abstract_final.csv"}

fenetre = tk.Tk()
fenetre.title("HarmonIA - Générateur de palettes de couleurs")
fenetre.geometry(f"{fenetre.winfo_screenwidth()}x{fenetre.winfo_screenheight()}")
fenetre.state("zoomed")  # Fonctionne sous Windows
#fenetre.attributes('-fullscreen', True)
fenetre.configure(bg="#ffffff")

styles = list(model_paths.keys())
style_selectionne = tk.StringVar(value="")

preds = ["Réseau de neurones", "Plus proche palette"]
pred_selectionnee = tk.StringVar(value="")

# Polices et couleurs
font_title = ("Bahnschrift", 25)
font_label = ("Helvetica Light", 12)
font_small = ("Bahnschrift Light", 7)
couleur_fond = "#ffffff"
couleur_fond_cadre = "#d9d9d9"
couleur_texte = "black"
couleur_bouton = "#626262"
couleur_bouton_hover = "#181818"

def distance_point(c_1, c_2):
    return np.linalg.norm(np.array(c_1) - np.array(c_2))

#Détermine la couleur d'affichage du texte
def texte_contraste(rgb):
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"

def color_width(input, pal_sep):
    """returns the proportions in width of each color block"""
    #distances = [[distance_point(input, prediction),i] for (i,prediction) in enumerate(pred_lab)]
    pal_sep = pal_sep[0]
    print(pal_sep)
    distances = np.array([distance_point(input, prediction) for prediction in pal_sep])
    print("Dist :", distances)
    inv_distances = 1/distances
    sum = np.sum(inv_distances)

    max_inv = np.max(inv_distances)
    inv_distances = np.concatenate([np.array([max_inv]), inv_distances])
    sum += max_inv
    print("Somme :",sum)
    print("Prop :",inv_distances/sum)

    return inv_distances/sum

# Récupère les couleurs issues de la prédiction par le perceptron
def get_colors(rgb, style):
    models = model_paths[style]
    model1 = load_model(models[0])
    model2 = load_model(models[1])
    model3 = load_model(models[2])

    lab = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    input = pd.DataFrame([[lab[0], lab[1], lab[2]]]).values

    pred1 = model1.predict(input)[0]
    pred_lab1 = np.array(pred1[:9])
    pal_sep1 = np.array([np.array([pred_lab1[0:3],pred_lab1[3:6],pred_lab1[6:9]])])
    pal_converting1 = np.clip(pal_sep1*255,0,255).astype(int)
    pal_converted1 = cv2.cvtColor(np.array(pal_converting1, dtype=np.uint8), cv2.COLOR_LAB2RGB)
    proportions1 = color_width(input,pal_sep1)

    pred2 = model2.predict(input)[0]
    pred_lab2 = np.array(pred2[:9])
    pal_sep2 = np.array([np.array([pred_lab2[0:3],pred_lab2[3:6],pred_lab2[6:9]])])
    pal_converting2 = np.clip(pal_sep2*255,0,255).astype(int)
    pal_converted2 = cv2.cvtColor(np.array(pal_converting2, dtype=np.uint8), cv2.COLOR_LAB2RGB)
    proportions2 = color_width(input,pal_sep2)

    pred3 = model3.predict(input)[0]
    pred_lab3 = np.array(pred3[:9])
    pal_sep3 = np.array([np.array([pred_lab3[0:3],pred_lab3[3:6],pred_lab3[6:9]])])
    pal_converting3 = np.clip(pal_sep3*255,0,255).astype(int)
    pal_converted3 = cv2.cvtColor(np.array(pal_converting3, dtype=np.uint8), cv2.COLOR_LAB2RGB)
    proportions3 = color_width(input,pal_sep3)

    # Couleurs à afficher
    couleurs1 = [
        rgb,
        pal_converted1[0][0],
        pal_converted1[0][1],
        pal_converted1[0][2],
    ]
    couleurs2 = [
        rgb,
        pal_converted2[0][0],
        pal_converted2[0][1],
        pal_converted2[0][2],
    ]
    couleurs3 = [
        rgb,
        pal_converted3[0][0],
        pal_converted3[0][1],
        pal_converted3[0][2],
    ]

    #couleurs = liste de 4 tuples en RGB de 0 à 255
    couleurs1 = [tuple(int(c) for c in couleur) for couleur in couleurs1]
    couleurs2 = [tuple(int(c) for c in couleur) for couleur in couleurs2]
    couleurs3 = [tuple(int(c) for c in couleur) for couleur in couleurs3]
    #print("Couleurs post tuple : ",couleurs)
    return [couleurs1, couleurs2, couleurs3, proportions1, proportions2, proportions3]


def activer_bouton(*args):
    val = style_selectionne.get()
    if val in styles:
        bouton.config(state="normal")
    else:
        bouton.config(state="disabled")

def afficher_pred_type(*args):
    val = pred_selectionnee.get()

def choisir_couleur():
    if style_selectionne.get() not in styles:
        return None
    
    if pred_selectionnee.get() == "Plus proche palette":

        df = pd.read_csv(csv_paths[style_selectionne.get()])
        df = df * 255  # Passage à l'échelle 0-255
        n_palettes = len(df)
        palettes = df.values.reshape((n_palettes, 4, 3))  # [n, 4, 3] en LAB
        all_colors = palettes.reshape((-1, 3))  # toutes les couleurs

        def affichage(rgb):
            lab = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0][0]
            R = 45  # rayon tolérance
            
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

        def find_palettes_from_color(input_lab_color, R, n):
            palettes_proches = []
            for i, color in enumerate(all_colors):
                dist = distance_point(color, input_lab_color)
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
        
        result = colorchooser.askcolor(title="Choisissez une couleur")
        rgb = result[0]

        if rgb:
            for widget in cadre_couleurs.winfo_children():
                widget.destroy()

            rgb_tuple = tuple(map(int, rgb))
            n, liste_palettes = affichage(rgb_tuple)

            if n == 0:
                label_aucune = tk.Label(cadre_couleurs, text="Aucune palette trouvée", font=font_label, background=couleur_fond_cadre, fg="black")
                label_aucune.pack()
                return

            # Affichage couleur choisie
            cadre_choisie = tk.Frame(cadre_couleurs, bg=couleur_fond_cadre, bd=2, padx=100, pady=10)
            cadre_choisie.pack(side="left", padx=10, pady=10)

            label = tk.Label(cadre_choisie, text="Couleur choisie", font=font_label, fg="white", bg=couleur_fond_cadre)
            label.pack(pady=(0, 6))

            hex_color = "#%02x%02x%02x" % rgb_tuple
            case = tk.Frame(cadre_choisie, bg=hex_color, width=120, height=120)
            case.pack()

            # Affichage des palettes trouvées
            for i, palette in enumerate(liste_palettes, start=1):
                cadre = tk.Frame(cadre_couleurs, bg=couleur_fond_cadre, padx=10, pady=0)
                cadre.pack(side="left", padx=10, pady=0)

                for couleur in palette:
                    hex_color = "#%02x%02x%02x" % couleur
                    case = tk.Frame(cadre, bg=hex_color, width=90, height=100)
                    case.pack(pady=0)

                    label_rgb = tk.Label(case, text=str(couleur), font=font_small, fg=texte_contraste(couleur), bg=hex_color)
                    label_rgb.place(relx=0.5, rely=0.5, anchor="center")


    # Si on a choisi le réseau de neurones
    if pred_selectionnee.get() == "Réseau de neurones" :
        result = colorchooser.askcolor(title="Choisissez une couleur")
        input_rgb = result[0]

        if input_rgb:
            for widget in cadre_couleurs.winfo_children():
                widget.destroy()

            input_rgb_tuple = tuple(map(int, input_rgb))
            liste_couleurs_prop = get_colors(input_rgb_tuple, style_selectionne.get())
            liste_couleurs = liste_couleurs_prop[:3]
            liste_prop = liste_couleurs_prop[3:]
            print(liste_prop)

            # Container for palette1
            palette_frame1 = tk.Frame(cadre_couleurs, bg=couleur_fond_cadre, pady=30)
            palette_frame1.pack(pady=0)

            for i, couleur in enumerate(liste_couleurs[0]):
                hex_color = "#%02x%02x%02x" % couleur

                # Create a vertical container for the color + label
                color_container = tk.Frame(palette_frame1, bg=couleur_fond_cadre)
                color_container.grid(row=0, column=i, padx=0, pady=0)

                # The color block
                case = tk.Frame(
                    color_container,
                    bg=hex_color,
                    width=color_w*liste_prop[0][i],   # width of each block
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

            # Container for palette2
            palette_frame2 = tk.Frame(cadre_couleurs, bg=couleur_fond_cadre, pady=30)
            palette_frame2.pack(pady=0)

            for i, couleur in enumerate(liste_couleurs[1]):
                hex_color = "#%02x%02x%02x" % couleur

                # Create a vertical container for the color + label
                color_container = tk.Frame(palette_frame2, bg=couleur_fond_cadre)
                color_container.grid(row=0, column=i, padx=0, pady=0)

                # The color block
                case = tk.Frame(
                    color_container,
                    bg=hex_color,
                    width=color_w*liste_prop[1][i],   # width of each block
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

            # Container for palette3
            palette_frame3 = tk.Frame(cadre_couleurs, bg=couleur_fond_cadre, pady=30)
            palette_frame3.pack(pady=0)

            for i, couleur in enumerate(liste_couleurs[2]):
                hex_color = "#%02x%02x%02x" % couleur

                # Create a vertical container for the color + label
                color_container = tk.Frame(palette_frame3, bg=couleur_fond_cadre)
                color_container.grid(row=0, column=i, padx=0, pady=0)

                # The color block
                case = tk.Frame(
                    color_container,
                    bg=hex_color,
                    width=color_w*liste_prop[2][i],   # width of each block
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


header_frame = tk.Frame(fenetre, bg=couleur_fond)
header_frame.pack(padx=10, pady=10, fill="x")

label_titre = tk.Label(
    header_frame,
    text="HarmonIA",
    font=font_title,
    fg="black",
    bg=couleur_fond,
)
label_titre.grid(row=0, column=0, sticky="w", padx=(10, 20))

label_sous_titre = tk.Label(
    header_frame,
    text="Le générateur de palettes harmonieuses",
    font=("Bahnschrift Light", 14),
    fg="black",
    bg=couleur_fond,
)
label_sous_titre.grid(row=1, column=0, sticky="w", padx=(10, 20))


# Conteneur pour menus + bouton
cadre_controls = tk.Frame(fenetre, bg=couleur_fond)
cadre_controls.pack(pady=15)

# Menu déroulant pred
cadre_menu_pred = tk.Frame(cadre_controls, bg=couleur_fond)
cadre_menu_pred.pack(side="left", padx=(0, 100), pady=(0,15))  # marge à droite

label_menu_pred = tk.Label(
    cadre_menu_pred,
    text="Mode de prédiction :",
    font=font_label,
    fg=couleur_texte,
    bg=couleur_fond,
)
label_menu_pred.grid(row=0, column=0, sticky="w")

menu = tk.OptionMenu(cadre_menu_pred, pred_selectionnee, *preds)
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


# Menu déroulant styles
cadre_menu_styles = tk.Frame(cadre_controls, bg=couleur_fond)
cadre_menu_styles.pack(side="left", padx=(0, 100), pady=(0,15))  # marge à droite

label_menu = tk.Label(
    cadre_menu_styles,
    text="Inspiration :",
    font=font_label,
    fg=couleur_texte,
    bg=couleur_fond,
)
label_menu.grid(row=0, column=0, sticky="w")

menu = tk.OptionMenu(cadre_menu_styles, style_selectionne, *styles)
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
    cadre_controls,
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

# Bouton à droite du menu déroulant
bouton.config(padx=100, pady=10)  # ajustement esthétique
bouton.pack(side="left")  # aligné à gauche dans cadre_controls



#bouton.pack(pady=15)


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
pred_selectionnee.trace_add("write", afficher_pred_type)

fenetre.mainloop()
