#from math import e, log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def DetermineDistribution(array):
    #-------separate data into classes, only works for given data set, should be separated based on class column
    classTwo = np.copy(array[0:25, 1:2])
    classThree = np.copy(array[25:50, 1:2])
    classOne = np.copy(array[50:75, 1:2]) 
    classZero =  np.copy(array[75:100, 1:2])
    
    
    #---------show histograms and 5 number summaries 
    #print(pd.DataFrame(classZero).describe(), end='\n\n')
    plt.hist(classZero, bins=30)
    plt.show()
    
   
    #print(pd.DataFrame(classOne) .describe(), end='\n\n')
    plt.hist(classOne, bins=30)
    plt.show()
    

    #print(pd.DataFrame(classTwo) .describe(), end='\n\n')
    plt.hist(classTwo, bins=30)
    plt.show()
    
    
    #print(pd.DataFrame(classThree) .describe(), end='\n\n')
    plt.hist(classThree, bins=30) 
    plt.show()

#---------------ANSWERS TO QUESTION 1----------------------------------------------------------
    #Class 0/Finch - Exponential 
    #Class 1/Duck - Uniform 
    #Class 2/Sparrow - Gaussian
    #Class 3/Raven - Exponential


def learnParams(data):
    
    #num classes in dataset
    classNum = len(np.unique(data[:, 0]))

    #zero array for parameters
    params = np.zeros((classNum, 2), dtype=float, order='C')


    for c in range(0, classNum, 1):
        elements = np.isin(data[:, 0], [c]) #find where given 'c' class is
        cData = data[elements] #get data for given class
        inst = len(cData[:, 0])  #get number of instances


        params[c][0] = 1/(np.mean(cData[:, 1], axis=0)) #calculate for each param, place in np array
        params[c][1] = 1/(np.mean(cData[:, 2], axis=0))
    
    print(f'params: {params}')
    return params


def learnPriors(data):
    
    #number of classes in data set
    classNum = len(np.unique(data[:, 0]))
    
    #----------total number of data points/rows and columns
    tRows, tCols = data.shape
    
    #empty array for priors to be put in
    priors = np.zeros((4))

    #get instance count for each class, divide by total number of observations
    for c in range(0, 4, 1):
        cRows, cCols = data[data[:, 0] == c].shape
        priors[c] = cRows/tRows

    return priors    


def labelBayes(birdFeats, params, priors):
    

    inst = len(birdFeats) #get k, number of birds to classify
    feats = len(birdFeats[0]) #number of featues
    numClasses = len(params) #number of classes
    

    labels = np.zeros((inst), dtype= int) #empty array for labels
    

    for bird in range(0, inst, 1): #for each bird observation, create a temporary list to hold likelihood of given class, take index of argmax as class to return
        tmpProb = np.zeros(numClasses)

        for c in range(0, numClasses, 1):
            
            tmpProb[c] = np.log(params[c, 0]) - (params[c, 0] * birdFeats[bird, 0])
            tmpProb[c] += np.log(params[c, 1]) - (params[c, 1] * birdFeats[bird, 1])
            tmpProb[c] += np.log(priors[c])

                #version with regular/non-log likelihood
                #params[c][f] * e**(-params[c][f] * birdFeats[bird][f])
                #tmpProb.append(prob + np.log(priors[c]))

        
        #print(f'TempProb Debug:\n{tmpProb}')
        labels[bird] = np.argmax(tmpProb)
       
    #print(f'Return Labels:\n{labels}')


    return labels



def main():
    
    inData = pd.read_csv("hw1Data.csv", header = None)

    inData = inData.to_numpy()
    #print(inData, end='\n\n')

    #divide into classes
    spdArr = np.copy(inData[:, 0:2])
    #print(spdArr)
    
    #make fake data array

    #DetermineDistribution(spdArr)
    params = learnParams(inData)
    priors = learnPriors(inData) 
    #labelsOut = labelBayes(np.array([[0.5,5],[0.5,2],[2,8]]), test data function call
    #                        np.array([[0.7,0.2],[0.4,0.1]]),
    #                        np.array([0.4,0.6]))
    labelTwo = labelBayes(inData[:, 1:3], params, priors)
    print(f'Full Dataset Labels:\n{labelTwo}')
    #print(f'Full Original Labels:\n{inData[:, 0]}')





if __name__ == "__main__":
    main()

