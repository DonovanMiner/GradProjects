import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import skimage.io
import copy

def Normalize_Features (img):
    
    divisor = np.full(shape = (np.shape(img)), fill_value = 255, dtype = float)
    #print(f"divisor array\n{divisor}")

    return img/divisor


def Compare_SSE (curr, prev, count):
     
    if count == 0 or count == 1:   #for initial loops
        return True
    elif prev == curr:
        return False
    else:
        return True



def K_Means (img, k_centroids):
    
    dimensions = np.shape(img)
    rows, columns, values = dimensions
    #print(f"dimensions of image: {dimensions}")
    cluster_classes = np.empty(shape= (rows, columns), dtype= int, order='C') 
    cluster_classes_prev = np.empty(shape= (rows, columns), dtype= int, order='C') 
    cluster_distances = np.empty(shape= (rows, columns), dtype= float, order='C')
    #print(f"empty classes:\n{cluster_classes}\nempty distances:\n{cluster_distances}")
    k_centroids_prev = []
    K_SSE = 0.0
    K_SSE_prev = 1.0 #need to be different nums for inital loop
    cont_iter = True
    #USE BOOL TO DETECT WHEN SSE INCREASES, PUT CHECK AT END OF FUNCTION, IF FALSE USE PREV KCLASSES, OTHERWISE USE CURRENT

    iter_count = 0
    while (cont_iter and Compare_SSE(K_SSE, K_SSE_prev, iter_count) and (iter_count < 50)):  
        
        cluster_classes_prev = copy.deepcopy(cluster_classes)
        K_SSE_prev = K_SSE
        #assign class based on least distance to centroids, maybe do with enumerate?
        for row in range(rows):
            for col in range(columns):    
                min_dist = (2**31)-1  #random high num to reassign to minimum distance to centroid later
                min_class = 0

                for centroid in range(len(k_centroids)):
                    euclid_dist = np.linalg.norm(img[row][col] - k_centroids[centroid])

                    if euclid_dist < min_dist:
                        min_dist = euclid_dist
                        min_class = centroid
                        #print(f"curr cent: {centroid}")
    
                    cluster_classes[row][col] = min_class
                    cluster_distances[row][col] = min_dist

        #print(f"CLUSTER CLASSES: {cluster_classes}\nCLUSTER DISTANCES: {cluster_distances}\n")


        #shifting centroid location
        #should be list of indexes of all items matching current centroid, use these indexes to shift through distance list and shift centroid location
        
        
        for centroid in range(len(k_centroids)):
            SSE = 0.0
            centroid_nodes = np.argwhere(cluster_classes == centroid)  #get indexes of given cluster "centroid"
            num_nodes = len(centroid_nodes)
            print(f"num of centroid nodes {num_nodes}")
            print(f"INDEXES OF CLUSTER {centroid}:\n{centroid_nodes}") 
            
            #calculate SSE for given centroid
            for i, indexes in enumerate(centroid_nodes):
                SSE += np.linalg.norm(img[indexes[0]][indexes[1]]- k_centroids[centroid])
                
            K_SSE = SSE

            #shifting centroids through mean values of cluster nodes 
            r_cen, g_cen, b_cen = 0.0, 0.0, 0.0
            for i, indexes in enumerate(centroid_nodes):
                
                pixel_RGB = img[indexes[0]][indexes[1]]
                #print(f"pixels: {pixel_RGB}")
                    
                r_cen += pixel_RGB[0]
                g_cen += pixel_RGB[1]
                b_cen += pixel_RGB[2]
            
            if (num_nodes > 0):
                r_cen = r_cen / num_nodes              
                g_cen = g_cen / num_nodes
                b_cen = b_cen / num_nodes
                k_centroids[centroid][0] = r_cen
                k_centroids[centroid][2] = g_cen
                k_centroids[centroid][1] = b_cen
            
        print(f"PREV CENTROIDS: {k_centroids_prev}\nCURR CENTROIDS: {k_centroids}")
        print(f"PREV SSE: {K_SSE_prev}\nCURR SSE: {K_SSE}")
        k_centroids_prev = copy.deepcopy(k_centroids)
        
        
        print(f"ITERATION {iter_count}\n\n")
        iter_count += 1

        if (K_SSE > K_SSE_prev and iter_count > 1):
            cont_iter = False
        
    
    if cont_iter == False:
        return k_centroids_prev, cluster_classes_prev
    else:
        return k_centroids, cluster_classes


def Recolor_Image(classes):
    
    row, col = np.shape(classes)
    colored_image = np.empty(shape= (row, col, 3), dtype= int, order='C')
    #print(f"colored image format:\n{colored_image}")
    #print(f"classes:\n{classes}")

    CLUSTER_COLORS = [[60, 179, 113], [0, 191, 255], [255, 255, 0], [255, 0, 0], [0, 0, 0], [169, 169, 169], [255, 140, 0], [128, 0, 128], [255, 192, 203], [255, 255, 255]]
          #colors are   Spring Green    DeepSkyBlue     Yellow         Red          Black       DarkGray        DarkOrange      Purple         Pink             White 

    for r in range(row):
        for c in range(col):
                clus_class = classes[r][c]
                #print(f"color is: {CLUSTER_COLORS[clus_class]}")
                colored_image[r][c] = CLUSTER_COLORS[clus_class]


    return colored_image




def SK_Color (centers, labels):
    
    #CLUSTER_COLORS = [[60, 179, 113], [0, 191, 255], [255, 255, 0], [255, 0, 0], [0, 0, 0], [169, 169, 169], [255, 140, 0], [128, 0, 128], [255, 192, 203], [255, 255, 255]]

    return centers[labels].reshape(244, 198, -1)




def main():
    
    #effiel tower pic
    img = skimage.io.imread("image.png")   #244x198x3 numpy array of RGB values
    skimage.io.imshow(img)
    plot.show()

    #print(f"raw RGB\n{img}")

    norm_img = Normalize_Features(img)
    #print(f"normalized data\n{norm_img}")

    k2 = [[0, 0, 0], [0.1, 0.1, 0.1]]
    k3 = [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
    k6 = [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5]]
    k10 = [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9]]

    
    k2_shifted, k2_classes = K_Means(norm_img, k2)  #returns centroids, cluster classes
    k2_img = Recolor_Image(k2_classes)
    skimage.io.imshow(k2_img)
    plot.show()
    #print(f"k2 rgb:\n{k2_img}\nOG format:\n{img}")
    #print(f"OG format type: {type(img)} and shape: {np.shape(img)}")  #debugs, RGB array formats
    #print(f"calced labels: {k2_classes}")
    #print(f"calced centers: {k2_shifted}")

    k3_shifted, k3_classes = K_Means(norm_img, k3)
    k3_img = Recolor_Image(k3_classes)
    skimage.io.imshow(k3_img)
    plot.show()

    k6_shifted, k6_classes = K_Means(norm_img, k6)
    k6_img = Recolor_Image(k6_classes)
    skimage.io.imshow(k6_img)
    plot.show()

    k10_shifted, k10_classes = K_Means(norm_img, k10)
    k10_img = Recolor_Image(k10_classes)
    skimage.io.imshow(k10_img)
    plot.show()


    return





if __name__ == "__main__":
    main()
