import cv2
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.pyplot as plt



def distance_point(c_1, c_2):
    # Conversion en [0,1]
    c1 = np.array(c_1) / 255
    c2 = np.array(c_2) / 255

    # Convertir RGB → Lab
    lab1 = color.rgb2lab([[c1]])[0][0]
    lab2 = color.rgb2lab([[c2]])[0][0]

    # Distance euclidienne dans l'espace Lab (≈ perceptuelle)
    distance = np.linalg.norm(lab1 - lab2)

    #affichage
    plt.imshow([[c_1,c_2]])
    plt.show()

    return distance

def distance_liste(cible_rgb, liste_rgb):
    # Mise à l'échelle [0, 1]
    cible_rgb = np.array(cible_rgb) / 255.0
    liste_rgb = np.array(liste_rgb) / 255.0

    # Conversion en Lab
    cible_lab = color.rgb2lab([[cible_rgb]])[0][0]       # (3,)
    liste_lab = color.rgb2lab(liste_rgb.reshape(-1, 1, 3))[:, 0, :]  # (N, 3)

    # Distance euclidienne (ΔE) entre la cible et chaque couleur
    distances = np.linalg.norm(liste_lab - cible_lab, axis=1)

    return distances


def foyer(n,M):
    '''Le but est de génerer n foyers le premier choisi au hasard, le deuxième chosi de sorte que la distance soit la plus loin du premier et itération suivante la plus loin des précédents, n nombre de foyers, M matrice des points'''
    M = np.array(M)
    premier_foyer = M[500]
    foyers = [premier_foyer]# Choisir le premier foyer au hasard

    # Distance initiale entre tous les points et le premier foyer
    distances_min = [hsv_distance(point, premier_foyer) for point in M]

    for i in range(1, n):
        idx_max = np.argmax(distances_min)
        prochain_foyer = M[idx_max]
        foyers.append(prochain_foyer)

        # Mettre à jour les distances minimales
        new_distances = [hsv_distance(point, prochain_foyer) for point in M]
        distances_min = np.minimum(distances_min, new_distances)

    print("Foyers choisis :", foyers)
    return np.array(foyers)

def moyenne(cluster):
    """Calcul le centre (couleur moyenne) d'un cluster."""
    if not cluster:
        return np.array([0, 0, 0])
    return np.mean(cluster, axis=0)

class KMeans:
    def __init__(self, n_clusters, max_iter=4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        # Array containing the modified image
        #modified_img = np.array(X_train, dtype=np.uint8)
        modified_img = np.full_like(X_train, [0,0,255])

        # Initialize centroids using the foyer function
        self.centroids = foyer(self.n_clusters, X_train)

        # Iterate until convergence or max iterations
        iteration = 0
        prev_centroids = None
        while iteration < self.max_iter and np.not_equal(self.centroids, prev_centroids).any() :
            # Assign points to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            sorted_points_coord = [[] for _ in range(self.n_clusters)]
            for i,x in enumerate(X_train):
                dists = []
                # Calculate distances to each centroid
                for center in self.centroids:
                    dists.append(hsv_distance(x, center))
                argmin = np.argmin(dists)
                centroid_idx = argmin

                sorted_points[centroid_idx].append(x)
                # Store the index of the point in the corresponding cluster to rebuild the image later
                sorted_points_coord[centroid_idx].append(i)

            # Update centroids
            prev_centroids = self.centroids
            self.centroids = np.array([moyenne(cluster) for cluster in sorted_points])          
            iteration += 1
        
        clusters = []
        clusters_coord = []
        clusters_centroids = []

        # Select 4 clusters with different colors
        # Select the most represented cluster for first one
        max = 0
        for i, cluster in enumerate(sorted_points):
            if len(cluster) > max:
                max = len(cluster)
                idx = i

        clusters.append(sorted_points[idx])  # Append the most represented cluster
        clusters_coord.append(sorted_points_coord[idx])  # Append the corresponding coordinates
        clusters_centroids = [self.centroids[idx]]

        # Select the clusters with the centroids that are the most distant from the first one
        for _ in range(3):
            distances = np.array([min([hsv_distance(new_centroid, centroid) for new_centroid in clusters_centroids]) for centroid in self.centroids])
            idx = np.argmax(distances)
            next_centroid = self.centroids[idx]

            clusters.append(sorted_points[idx])
            clusters_coord.append(sorted_points_coord[idx])
            clusters_centroids.append(next_centroid)

        self.centroids = clusters_centroids

        # Rebuild the image
        for i, cluster in enumerate(clusters):
            for idx in clusters_coord[i]:
                modified_img[idx] = self.centroids[i]

        modified_img = modified_img.reshape((100, 100, 3)).astype(np.uint8)
        plt.imshow(cv2.cvtColor(modified_img, cv2.COLOR_HSV2RGB))
        
        print(f"Convergence après {iteration} iterations.")
        dico = {}
        for i, cluster in enumerate(clusters):
            dico[i] = [self.centroids[i], len(cluster)]
        print("Dictionnaire des clusters:", dico)
        return dico



#idealement renvoie d[couleur] = population

#exemple : 

img = cv2.imread('compressed_images_hsv/50.png', cv2.IMREAD_COLOR)

#print(img.reshape(-1, 3).shape)  # Reshape the image to a 2D array of pixels
kmeans = KMeans(n_clusters=8).fit(img.reshape(-1, 3))

#convert the clusters from HSV to RGB
for i, (centroid, population) in kmeans.items():
    centroid = np.clip(centroid, 0, 255).astype(int)
    kmeans[i] = (cv2.cvtColor(np.array([[centroid]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0], population)

#Plot the clusters
plt.figure(figsize=(10, 5))
for i, (centroid, population) in kmeans.items():
    plt.bar(i, population, color=centroid / 255, label=f'Cluster {i} (Population: {population})')
plt.xlabel('Cluster')
plt.ylabel('Population')
plt.title('Population of Clusters')
plt.legend()
plt.show()