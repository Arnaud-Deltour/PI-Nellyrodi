import cv2
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
img = cv2.imread('PI-Nellyrodi/image/IMG_4697.png')

hauteur, largeurs, couleurs = img.shape

def distance (x,y):
  
    '''distance euclidienne entre x et y'''
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)**0.5



def trouver_centre (L):
    centre = L[0]
    s = 0
    for y in L:
        s += distance(centre,y)
    m=s
    for index,x in enumerate(L):
        s = 0
        for y in L:
            s += distance(x,y)
        if s < m:
            m = s
            centre = x
    return centre

def trouver_barycentre (L):
    '''trouve le barycentre de la liste L'''
    s = [0,0,0]
    for x in L:
        s[0] += x[0]
        s[1] += x[1]
        s[2] += x[2]
    return [s[0]/len(L),s[1]/len(L),s[2]/len(L)]

def clusters (Im,L,C,N):
    '''L la liste des foyers, M matrice des clusters, Im le tableau de l'image, C affichage des clusters,itérations'''
    for k in range(N):
        M = [[Im[0,0]] for i in range(len(L))]
        for j in range (hauteur):
            print('changement hauteur')
            for index,x in enumerate(Im[j]) :
                d = distance(x,L[0])
                for i in range(1,len(L)):
                    d2 = distance(x,L[i])
                    
                    if d2 < d:
                        d = d2
                        c = i
                M[c].append(x)
            
        for i in range(len(L)):
            print('recherche barycentre')
            L[i] = trouver_barycentre(M[i])
            C[j][index] = L[i]   
    return(L,M,C)

L = [img[rd.randint(0,hauteur)][rd.randint(0,largeurs)] for i in range(5)]
L,M,C = clusters(img,L,[[[0,0,0] for i in range (largeurs)] for k in range (hauteur)],10)

plt.imshow(C)
plt.show()



        
        




        
            



    


    

  




    


        




