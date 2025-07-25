import cv2
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt

# img = cv2.imread('PI-Nellyrodi/data_hsv/impressionist_paintings/2019.jpg', cv2.IMREAD_COLOR)
img = cv2.imread("data/abstract_lab/88.png", cv2.IMREAD_COLOR)
# img = cv2.imread('abstract_lab/4360.png', cv2.IMREAD_COLOR)
# img = cv2.imread('image/img.jpg', cv2.IMREAD_COLOR)


def foyer(n, M):
    """Le but est de générer n foyers :
    - Le premier est choisi arbitrairement (ici l’élément à l’index 500),
    - Chaque suivant est le plus éloigné (en Lab) des foyers précédents."""

    M = np.array(M)
    premier_foyer = M[500]
    foyers = [premier_foyer]

    # Distance initiale entre tous les points et le premier foyer
    distances = np.array([distance_point(premier_foyer, point) for point in M])

    for i in range(1, n):
        idx_max = np.argmax(distances)
        prochain_foyer = M[idx_max]
        foyers.append(prochain_foyer)

        # Mettre à jour les distances minimales (par rapport à l'ensemble des foyers)
        new_distances = np.array([distance_point(prochain_foyer, point) for point in M])
        distances = np.minimum(distances, new_distances)

    print("Foyers choisis :", foyers)
    return np.array(foyers)


def distance_point(c_1, c_2):
    return np.linalg.norm(c_1 - c_2)


def moyenne(cluster):
    """Calcul le centre (couleur moyenne) d'un cluster."""
    if not cluster:
        return np.array([0, 0, 0])
    return np.mean(cluster, axis=0)


def dist_transform(distances):
    return distances


class KMeans:
    def __init__(
        self,
        n_clusters,
        n_clusters_init=8,
        max_iter=4,
        demo=False,
        print_clusters=False,
    ):
        self.n_clusters = n_clusters
        self.n_clusters_init = n_clusters_init
        self.max_iter = max_iter
        self.demo = demo
        self.print_clusters = print_clusters

    def fit(self, X_train):
        if self.demo:
            # Array containing the modified image
            modified_img = np.full_like(X_train, [255, 127, 127])
            modified_img_totale = np.full_like(X_train, [255, 127, 127])

        # Initialize centroids using the foyer function
        self.centroids = foyer(self.n_clusters_init, X_train)

        # Iterate until convergence or max iterations
        iteration = 0
        prev_centroids = None
        while (
            iteration < self.max_iter
            and np.not_equal(self.centroids, prev_centroids).any()
        ):
            # Assign points to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters_init)]
            sorted_points_coord = [[] for _ in range(self.n_clusters_init)]
            for i, x in enumerate(X_train):
                dists = []
                # Calculate distances to each centroid
                for center in self.centroids:
                    # dists.append(hsv_distance(x, center))
                    dists.append(distance_point(x, center))
                argmin = np.argmin(dists)
                centroid_idx = argmin

                sorted_points[centroid_idx].append(x)
                if self.demo:
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

        idx_list = [idx]  # List of already selected indexes
        clusters.append(
            max
        )  # Append the number of points in the most represented cluster
        if self.demo:
            clusters_coord.append(
                sorted_points_coord[idx]
            )  # Append the corresponding coordinates
        clusters_centroids = [self.centroids[idx]]

        # Select the clusters with the centroids that are the most distant from the previous ones
        for _ in range(self.n_clusters - 1):
            distances = []
            for new_centroid in clusters_centroids:
                # distances.append(np.array([hsv_distance(new_centroid, centroid) for centroid in self.centroids]))
                distances.append(
                    np.array(
                        [
                            distance_point(new_centroid, centroid)
                            for centroid in self.centroids
                        ]
                    )
                )
            # On fait ensuite la moyenne des sous-tableaux de distances
            # print(distances)

            distances = np.array(distances)
            distances = dist_transform(distances)

            distances_moy = np.mean(np.array(distances), axis=0)

            for idx in idx_list:
                distances_moy[idx] = 0

            # print("dist moy :", distances_moy)

            # On en extrait le max
            idx = np.argmax(distances_moy)
            idx_list.append(idx)
            # print("idx :", idx)

            # On ajoute les points au cluster correspondant
            clusters.append(len(sorted_points[idx]))
            clusters_centroids.append(self.centroids[idx])

            if self.demo:
                clusters_coord.append(sorted_points_coord[idx])

        if self.demo:
            # Rebuild the whole image
            for i, cluster in enumerate(sorted_points):
                for idx in sorted_points_coord[i]:
                    modified_img_totale[idx] = self.centroids[i]

            modified_img_totale = modified_img_totale.reshape((100, 100, 3)).astype(
                np.uint8
            )
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cv2.cvtColor(modified_img_totale, cv2.COLOR_LAB2RGB))
            ax1.title.set_text(str(self.n_clusters_init) + " initial clusters")

            # Rebuild the image with selected clusters only
            self.centroids = clusters_centroids
            for i, cluster in enumerate(clusters):
                for idx in clusters_coord[i]:
                    modified_img[idx] = self.centroids[i]

            modified_img = modified_img.reshape((100, 100, 3)).astype(np.uint8)
            ax2.imshow(cv2.cvtColor(modified_img, cv2.COLOR_LAB2RGB))
            ax2.title.set_text(str(self.n_clusters) + " selected clusters")

        self.centroids = clusters_centroids
        print(f"Convergence après {iteration} iterations.")

        dico = {}
        for i in range(len(clusters)):
            dico[i] = [self.centroids[i], clusters[i]]

        if self.print_clusters:
            print("Dictionnaire des clusters:", dico)

            rgb_dico = {}

            # convert the clusters from LAB to RGB
            for i, (centroid, population) in dico.items():
                rgb_centroid = np.clip(centroid, 0, 255).astype(int)
                # kmeans[i] = (cv2.cvtColor(np.array([[centroid]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0], population)
                rgb_dico[i] = (
                    cv2.cvtColor(
                        np.array([[rgb_centroid]], dtype=np.uint8), cv2.COLOR_LAB2RGB
                    )[0][0],
                    population,
                )

            #   Plot the clusters
            plt.figure(figsize=(10, 5))
            for i, (centroid, population) in rgb_dico.items():
                plt.bar(
                    i,
                    population,
                    color=centroid / 255,
                    label=f"Cluster {i} (Population: {population})",
                )
            plt.xlabel("Clusters")
            plt.ylabel("Population")
            plt.title("Population of Clusters")
            plt.legend()

        plt.show()
        return dico


kmeans = KMeans(n_clusters=4, demo=True, print_clusters=True).fit(img.reshape(-1, 3))
