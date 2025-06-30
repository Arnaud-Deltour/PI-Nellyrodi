import cv2

def trouver_centre (L):
    centre = L[0]
    s = 0
    for y in L:
        s += distance(x,y)
    m=s
    for index,x in enumerate(L):
        s = 0
        for y in L:
            s += distance(x,y)
        if s < m:
            m = s
            centre = x
    return centre

def clusters (Im,L,C,N):
    '''L la liste des foyers, M matrice des clusters, Im le tableau de l'image, C affichage des clusters,itérations'''
    for k in range(N):
        M = [[] for i in range(len(L))]
        for index,x in enumerate(Im) :
            d = distance(x,L[0])
            for i in range(1,len(L)):
                d2 = distance(x,L[i])
                if d2 < d:
                    d = d2
                    c = i
            M[c].append(x)
            
        for i in range(len(L)):
            L[i] = trouver_centre(M[i])
    C[index] = L[i]
    return(L,M,C)




        
        




        
            



    


    

  




    


        




