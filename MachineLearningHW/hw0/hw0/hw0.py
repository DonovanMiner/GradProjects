import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as plt


def neighborClassify(featureArray, trainArray) -> list[int]:
    #takes in 1d numpy array (heights of uncalssified animals)
    #2nd arg is 2d numpy array (heights of classified animals)
    #return a list, 0 for non-giraffe, 1 for a giraffe
    
    giraffeClasses = []
    
    for f in range(len(featureArray)):
        minDist = float('inf')
        minIdx = 0
        for t in range(len(trainArray)):
            currDist = abs(featureArray[f] - trainArray[t][0])
            #print(f'currDist: {currDist}') Debug
            if(currDist < minDist):
                minDist = currDist
                minIdx = t
                #print(f'Updated Dist: {minDist}') 
                #print(f'Updated Index: {minIdx}')
        #print('\n\n\n')
        giraffeClasses.append(int(trainArray[minIdx][1]))

    print(f'Classified:\n {giraffeClasses}\n')    
    
    return giraffeClasses


def recalls(classifierOutput, trueLabels):
    
    oneTotal = 0
    oneTrue = 0
    zeroTotal = 0
    zeroTrue = 0
    
    for c, t in zip(classifierOutput, trueLabels):

        if(t == 0):
            if(t == c):
               zeroTotal += 1
               zeroTrue += 1
            else:
                zeroTotal += 1
        else:
            if(t == c):
                oneTotal += 1
                oneTrue += 1
            else:
                oneTotal += 1
        

    return np.array([zeroTrue/zeroTotal, oneTrue/oneTotal])


def removeOnes(dataArray):
    
    toDelete = []

    for i in range(len(dataArray)):
        if(dataArray[i][1] == 1):
            toDelete.append(i)
            
    returnArray = np.delete(dataArray, toDelete, axis = 0) 
    return returnArray




def main():
    
    #exampleArray = np.random.uniform(low = 0.1, high = 10.0, size = 10)
    exampleArray = np.array([6,3,9])
    #trainFeat = np.random.uniform(low = 0.1, high = 10, size = 10)
    trainArray=np.array([[0.5,0], [1.5,0], [2.5,0], [4.5,1], [5,1], [7.5, 0], [8,1], [9.2,1]]) 
    
    #trainClass = np.random.randint(low = 0, high = 2, size = (1, 10), dtype = int)
    #trainArray = np.concatenate((trainFeat.T, trainClass.T), axis = 1)
    print(f'Example:{exampleArray}\n')
    #print(f'Feats:\n {trainFeat}\n')
    #print(f'Classes:\n {trainClass}\n')
    print(f'Array:\n{trainArray}\n')
    neighborClassify(exampleArray, trainArray)
    
    classifierOutput=[0,1,1,0,0,1,1,0,1]
    trueLabels=[1,1,0,0,0,0,1,1,1]

    recallOutput = recalls(classifierOutput, trueLabels)
    print(f'Recall Vals:\n{recallOutput}')
    
    dataArray = np.array([[-4,2],[5,0],[3,0],[8,1],[10,0],[4,0],[2,1],[-2,2]])
    expandedData = removeOnes(dataArray)
    print(f'Removed Ones:\n{expandedData}')




if __name__ == "__main__":
    main()
