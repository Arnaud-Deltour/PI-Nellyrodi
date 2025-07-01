import tkinter as tk
from tkinter import colorchooser


def affichage(rgb):
    """
    Simule une fonction qui prend une couleur RGB et renvoie :
    - un entier n (nombre de cases)
    - une liste de n couleurs RGB (variations simulées ici)
    """
    r, g, b = rgb
    couleurs = []
    n = 5  # Par exemple, 5 couleurs à afficher

    # Générer n variations légères de la couleur d'origine
    for i in range(n):
        variation = (min(r + i * 20, 255), max(g - i * 10, 0), min(b + i * 15, 255))
        couleurs.append(variation)

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
