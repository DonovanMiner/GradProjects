import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import math
import json
import copy
from collections import Counter



def Euclidean_Distance(df_train, df_test):

    distances = {k + 1:{keys + 1:0 for keys in range(len(df_train.index))} for k in range(len(df_test.index))}
    #print(distances)

    for test_index in range(len(df_test.index)):
        x_test = df_test.iloc[test_index].to_numpy()

        for train_index in range(len(df_train.index)):
            x_train = df_train.iloc[train_index].to_numpy()
            diff = np.subtract(x_test, x_train)
            squared = np.power(diff, 2)
            square_sums = np.sum(squared, dtype=float)
            e_dist = math.sqrt(square_sums)
            distances[test_index+1][train_index+1] = (e_dist)


    #print(distances)

    return distances


def KNearest (distances, k):
    #take in dict, for every data instance in the dict, sort through its ED's k numbver of times retrieving the min val then deleting that minimum 

    dist = copy.deepcopy(distances) 
    min_dist = {k+1:{} for k in range(len(distances))}

    for test_val in range(len(dist)):
        for i in range(k):
            curr_min = list(min(dist[test_val + 1].items(), key= lambda x: x[1]))
            min_dist[test_val + 1].update({curr_min[0] : curr_min[1]}) 
            del dist[test_val + 1][curr_min[0]]
            
    return min_dist


def ZScoreNormalization (df_test):

    df_test_normalized = np.matrix(df_test.to_numpy())
    #print(f"print df_test to check dimensions:\n{df_test_normalized}")
    dimensions = np.shape(df_test_normalized)
    rows, columns = dimensions
    #print(f"rows, cols: {rows} {columns}\n")

    #means
    feat_means = (np.matrix.sum(df_test_normalized, axis=0))
    feat_means = feat_means / rows
    #print(f"Feature means:\n{feat_means}")


    #calc SD, for each instance in each column:
    #take difference from mean
    #square difference (sum these squared differences)
    #divide by number of instances (rows)
    #take square root  
    stand_devs = []
    for feature in range(columns):
        curr_feature = df_test_normalized[:, feature]
        
        SD_sum = 0
        for instance in range(rows):
            #print(f"feature val: {curr_feature[instance]}\nmean val: {feat_means[0, feature]}")
            dist_from_mean = curr_feature[instance] - feat_means[0, feature]
            SD_sum += dist_from_mean * dist_from_mean
        
        SD_sum = SD_sum / rows
        SD_sum = math.sqrt(SD_sum)
        #print(f"SD_sum: {SD_sum}")
        stand_devs.append(SD_sum)
        #print(f"list: {stand_devs}")



    #normailzation
    for feature in range(columns):


        for instance in range(rows):
            curr_val = df_test_normalized[instance, feature]
            #print(f"current value to edit: {curr_val}")
            curr_val -= feat_means[0, feature]
            curr_val /= stand_devs[feature]
            #print(f"current value to place: {curr_val}")
            df_test_normalized[instance, feature] = curr_val
            
        
    df_test_normalized = pd.DataFrame(df_test_normalized)
    return df_test_normalized


 
def Classify (min_distances, df_train):
    
    k_nearest = {k+1:[] for k in range(len(min_distances))}
    #print(f"k_nearest:\n{k_nearest}\n\n")
    #get classes
    for instance, minimums in min_distances.items():
        #print(f'Current data point: {instance}\n')
        for index, distance in min_distances[instance].items():
            k_nearest[instance].append(df_train[index - 1])
            #print(f"\tIndex of min value: {index}\n\tClass of index: {df_train[index - 1]}")
            #print(f"\tK-nearest result: {k_nearest[instance]}\n\n")


    #count k nearest
    for instance, classes in k_nearest.items():
        det_class = Counter(classes)
        det_class = max(det_class, key=det_class.get)
        k_nearest[instance] = det_class

    return k_nearest


def TestAccuracy (df_act, pred):
    
    total = len(df_act.index)

    sum_correct = 0
    for index in range(len(df_act.index)):
        if (pred[index + 1] == df_act[index]):
            sum_correct += 1

    return sum_correct/total


def main(): 
    df_spam_train = pd.read_csv("spam_train.csv", usecols= lambda c: not c.startswith('Unnamed:'))
    spam_train_feat = df_spam_train.iloc[:, 0:len(df_spam_train.columns) -1]
    spam_train_class = df_spam_train["class"]

    df_spam_test = pd.read_csv("spam_test.csv", usecols= lambda c: not c.startswith('Unnamed:'))
    spam_test_feat = df_spam_test.iloc[:, 1:len(df_spam_test.columns) - 1]
    spam_test_class = df_spam_test["Label"]    

    #USED TO CALCULATE ENSEMBLE METHOD C(25)--------------------------------------------------------------------------------------
    inacc_sum = 0.0
    n = 25
    k = math.ceil(25/2)
    print(f"test (25, 13): {math.comb(n, k)}")
    for i in range(k):
        #print(f"before: {k}")
        inacc_sum += math.comb(n, k) * math.pow(0.55, k) * math.pow(0.45, n - k)
        k +=1
        #print(f"after: {k}\n\n")
    print(f"Inaccuracy of 25 test: {inacc_sum}")#----------------------------------------------------------------------------------

    #print(f"df_train classes:\n{spam_train_class}")

    #print(f"Training set:\n{df_spam_train}")
    #print(f"Testing set:\n{df_spam_test}")

    #print(f"Testing X:\n{spam_test_feat}")
    #------------------------------------------------------------------------------------------------   
    #compare test instance to all the training instances, store k number of nearest values, determine what majority of nearest classes are
    #function to calculate euclidean distances between test instances and all train instances, store distances as {instance 1 : [distances], instance 2 : [distances]}
    #function that takes distance dictionary and retrieves the k number of lowest distances (be sure to make a copy of the dictionary to make it "pass by value") 

    #non normalized feature distances
    ALL_DIST = Euclidean_Distance(spam_train_feat, spam_test_feat)

    df_test_feat_zscore = ZScoreNormalization(spam_test_feat)
    df_train_feat_zscore = ZScoreNormalization(spam_train_feat)

    #print(f"normalized test features:\n{df_test_feat_zscore}\n\nnormalized training features:\n{df_train_feat_zscore}")

    #normalized feature distances
    Z_ALL_DIST = Euclidean_Distance(df_train_feat_zscore, df_test_feat_zscore)

    k_vals = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
    test_accuracies = {}
    for k_to_use in range(len(k_vals)):

        min_dist = KNearest(ALL_DIST, k_vals[k_to_use])         #find k number of nearest values 
        #print(f"\n\nMinimum distances:\n{min_dist}")

        k_near_classes = Classify(min_dist, spam_train_class) #classify based on k nearest values
        #print (f"My classes:\n {k_near_classes}")
    
        accuracy = TestAccuracy(spam_test_class, k_near_classes) #report test accuracy
        #print(f"test accuracy: {accuracy}")
        test_accuracies.update({k_vals[k_to_use] : accuracy})

         
    z_test_accuracies = {}
    for k_to_use in range(len(k_vals)):

        z_min_dist = KNearest(Z_ALL_DIST, k_vals[k_to_use])

        z_k_near_classes = Classify(z_min_dist, spam_train_class)
        print (f"Normalized classes for k{k_vals[k_to_use]}: {z_k_near_classes}")

        with open("norm_classes.txt", "a") as file:
            file.write(f"Normalized classes for k{k_vals[k_to_use]}:\n{z_k_near_classes}\n\n\n")

        z_accuracy = TestAccuracy(spam_test_class, z_k_near_classes)
        z_test_accuracies.update({k_vals[k_to_use] : z_accuracy})

    #console output
    print("Non-normalized:\n")
    for k, acc in test_accuracies.items():
        print(f"k value: {k}\ttest accuracy: {acc}")

    print("\n\nZ-Score\n")
    for k, acc in z_test_accuracies.items():
        print(f"K value: {k}\tTest Accuracy: {acc}")


    #with open("output.txt", "w") as file: #write distances
    #    file.write(json.dumps(ALL_DIST))

    #with open("z_output.txt", "w") as file: #write distances
    #    file.write(json.dumps(Z_ALL_DIST))

    #with open("output.txt", 'r') as file: #read distances
    #    text_distances = file.read()

    #with open("min_vals.txt", "w") as file: #write min values
    #    file.write(json.dumps(min_dist))

    return



if __name__ == "__main__":
    main()
