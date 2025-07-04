# HarmonIA – Générateur de palettes de couleurs artistiques

**HarmonIA** est une application interactive en Python (Tkinter) qui génère des palettes de couleurs harmonieuses à partir d'une couleur choisie. Elle propose deux approches complémentaires :

- Générer une palette via une **IA entraînée** sur des œuvres artistiques.
- Rechercher dans un **jeu de données existant** la palette la plus proche contenant une couleur similaire.

---

## Objectif

Permettre aux utilisateurs de créer ou explorer des palettes de couleurs inspirées de mouvements artistiques comme l’**art impressionniste** ou l’**art abstrait**, à partir d’une couleur choisie.

---

## Deux modes de génération

### 1. IA par réseau de neurones

- Trois réseaux de neurones sont disponibles pour chaque style artistique.
- Ces modèles ont été entraînés sur des palettes extraites d'images de peinture (`data_impressionist.csv` et `data_abstract_final.csv`).
- Donnée d’entrée : une couleur (choisie via colorpicker).
- En sortie : trois palettes de couleurs générées (1 couleur choisie + 3 suggestions), avec des **proportions calculées** selon la proximité dans l’espace colorimétrique.

---

### 2. Recherche de la palette la plus proche

- On parcourt un **dataset de palettes existantes**.
- On convertit la couleur choisie en espace **LAB**.
- On sélectionne les palettes du dataset contenant une couleur proche (selon une distance euclidienne en LAB).
- Résultat : les **palettes originales issues d’œuvres réelles** les plus proches sont affichées.


---

## Génération de Palettes de Couleurs Représentatives à partir d’Images - principe général

Le script clusters.py script extrait automatiquement des **palettes de 4 couleurs distinctes** à partir d’images existantes, afin de constituer un dataset de palettes harmonieuses.


### 1. Compression homogène des images (100x100 px)

- Chaque image est réduite à une taille standard de 100x100 pixels **sans interpolation**, en ne conservant que des pixels réellement présents dans l’image d’origine.
- Cela garantit que les couleurs extraites sont authentiques et représentatives.


### 2. Transformation en espace couleur LAB

- Le traitement des couleurs se fait dans l’espace **LAB**, plus adapté à la perception humaine que RGB ou HSV.
- La distance entre deux couleurs est mesurée par la distance euclidienne dans cet espace, reflétant mieux les différences visuelles.


### 3. Clustering personnalisé (KMeans modifié)

- Un algorithme KMeans identifie d’abord **8 clusters de couleurs dominantes** dans l’image.
- Parmi ces 8 clusters, on sélectionne **4 couleurs finales** selon ces critères :
  - Le premier cluster est la couleur la plus fréquente.
  - Les 3 autres sont choisis pour être **le plus éloignés possible en couleur** des précédents, assurant diversité et contraste.
- Cette méthode produit une palette finale variée et cohérente.


### 4. Sauvegarde dans un fichier CSV

- Les 4 couleurs sélectionnées sont enregistrées, avec leur fréquence respective, dans un fichier `.csv`.
- Ce dataset sert ensuite pour entraîner des modèles ou alimenter des interfaces graphiques.