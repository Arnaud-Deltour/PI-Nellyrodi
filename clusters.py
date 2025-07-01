import cv2
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
img = cv2.imread('data_hsv/impressionist_paintings/0.jpg', cv2.IMREAD_COLOR)



def hsv_distance(p1, p2):  # p1 et p2 sont des triplets de la forme [h,s,v]
    r1 = (p1[1] / 255) * (p1[2] / 255)
    theta1 = (p1[0] / 180) * 2 * np.pi
    z1 = p1[2] / 255 - 1
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    r2 = (p2[1] / 255) * (p2[2] / 255)
    theta2 = (p2[0] / 180) * 2 * np.pi
    z2 = p2[2] / 255 - 1
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def moyenne(cluster):
    """Calcul le centre (couleur moyenne) d'un cluster."""
    if not cluster:
        return np.array([0, 0, 0])
    return np.mean(cluster, axis=0)

def initialize(data, n):
    centroids = []
    centroids.append(data[np.random.randint(data.shape[0])])

    for _ in range(n - 1):
        distances = []
        for point in data:
            min_dist = min([hsv_distance(point, c) for c in centroids])
            distances.append(min_dist)
        
        next_centroid = data[np.argmax(distances)]
        centroids.append(next_centroid)
    
    return np.array(centroids)

class KMeans:
    def __init__(self, n_clusters, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        # Initialize centroids
        # Run initialization
        self.centroids = initialize(X_train, n=self.n_clusters)
        #min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        #self.centroids = [np.random.uniform(min_, max_) for _ in range(self.n_clusters)]

        # Iterate until convergence or max iterations
        iteration = 0
        prev_centroids = None
        while iteration < self.max_iter and np.not_equal(self.centroids, prev_centroids).any() :
            # Assign points to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = []
                # Calculate distances to each centroid
                for center in self.centroids:
                    dists.append(hsv_distance(x, center))
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Update centroids
            prev_centroids = self.centroids
            self.centroids = np.array([moyenne(cluster) for cluster in sorted_points]) 
            print(f"Iteration {iteration + 1}")         
            iteration += 1
        print(f"Convergence après {iteration} iterations.")
        dico = {}
        for i, cluster in enumerate(sorted_points):
            dico[i] = [self.centroids[i], len(cluster)]
        print("Dictionnaire des clusters:", dico)
        return dico

"""

#idealement renvoie d[couleur] = population

#print(img.reshape(-1, 3).shape)  # Reshape the image to a 2D array of pixels
kmeans = KMeans(n_clusters=7).fit(img.reshape(-1, 3))

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

"""