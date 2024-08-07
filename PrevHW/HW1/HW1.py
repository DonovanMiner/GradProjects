import pandas as pd
import matplotlib.pyplot as plot
import numpy as np




def GetX(df):
    x = df.iloc[:, 0:len(df.columns) - 1]
    #print(f"Whole x matrix:\n{x}\n") #debug

    return x

def GetXNaught(df):
    x = df.iloc[:, 0:len(df.columns) - 1]
    x.insert(0, "x(0)", 1.0)
    
    #print(f"X matrix with 1s:\n{x}\n") #debug

    return x

def GetYVec(df):
    y = df["y"]
    #print(f"Y/Target vector:\n{y}\n")

    return y

def GetW_WithNaught(x, y):
    
    xT = np.matrix.transpose(x)
    xT_x = np.matmul(xT, x)
    xT_x_inv = np.linalg.inv(xT_x)
    x_dag = np.matmul(xT_x_inv, xT)
    w_vals = np.matrix(np.matmul(x_dag, y))
    w = pd.DataFrame(w_vals)

    #print(w) #debug

    return w

def GetMSE(x, y, w, w_0):
    #x in matrix
    #y in vertical vector
    #w in horizontal vector
    #w(0) is a value
   
    x_np = x.to_numpy()
    w_np = w.to_numpy()
    y_np = y.to_numpy()
    w_0_np = w_0.to_numpy()

    #print(f"x:\n{x_np}\n")
    #print(f"w:\n{w_np}\n")
    #print(f"y:\n{y_np}\n")
    #print(f"w(0): {w_0_np}\n")

    #MSE/E(w) = ||Xw - y||^2 / N  subtraction?
    
    MSE_sum = 0.0
    for i in range(len(x.index)):
        predicted = w_0_np + np.matmul(w_np, x_np[i])
        actual = y_np[i]
        diff = predicted -actual    
        MSE_sum += pow(diff, 2)

        #print(f"--------{i}--------")
        #print(f"Predicted: {predicted}")
        #print(f"Actual: {actual}")
        #print(f"Difference: {diff}")
        
    MSE = MSE_sum / len(x.index)

    return MSE

def GetMSEMatrix(y_pred, y_act):
   
    diff = np.subtract(y_pred, y_act)
    #print(f"diff:\n{diff}\n")

    diff_squared = np.matmul(diff.T, diff)
    #print(f"diff squared:\n{diff_squared}\n")
    MSE = diff_squared / len(y_pred)

    return MSE
     
def GetPredictedValues(x, w, w_0):
    #create empty python list, fill in with predicted values, convert to pandas dataframe
    
    x_np = x.to_numpy()
    w_np = w.to_numpy()
    w_0_np = w_0.to_numpy()
    
    predicted = []
    for i in range(len(x.index)):
        y = w_0_np + np.matmul(w_np, x_np[i])
        predicted.append(y)
        

    y_pred = pd.DataFrame(predicted)
    #print(f"Predicted list:\n{predicted}\n")
    #print(f"Predicted values:\n{y_pred}\n")

    return y_pred

def GetL2Norm(x, y, w, w_0):
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    w_np = w.to_numpy()
    w_0_np = w_0.to_numpy()

    L2 = 0.0
    for i in range(len(x.index)):
        predicted = w_0_np + np.matmul(w_np, x_np[i])
        actual = y_np[i]
        diff = predicted -actual    
        L2 += pow(diff, 2)

    return L2

def NonLinearTransformation(x, w, complexity):
    
    #print(f"w base test:\n{w}\nshape: {w.shape}") #debugs
    #print(f"x base test:\n{x}\nshape: {x.shape}")
    
    w_0 = w[0][0]
    w = np.delete(w, 0, axis= 0)
    x = np.delete(x, 0, axis= 1)
    #print(f"w edited test:\n{w}\nshape: {w.shape}")
    #print(f"x edited test:\n{x}\nshape: {x.shape}")


    y_NLT_pred = np.zeros((len(x), 1), dtype=float)
    for c in range(1, complexity+1, 1):
        x_pow = np.power(x, c)
        #print(f"x pow:\n{x_pow}\nshape: {x_pow.shape}") #debug
        c_sum =  np.matmul(x_pow, w)
        y_NLT_pred = np.add(y_NLT_pred, c_sum)

    
    y_NLT_pred = y_NLT_pred + w_0
    #print(f"base y_NLT_pred:\n{y_NLT_pred}") #more debugs
    nlt_pred = np.array(y_NLT_pred.T)[0]
    #print(f"pls work:\n{nlt_pred}")


    return nlt_pred


def MinLambdaValue(x, y, start_lam, end_lam, complexity):
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    x_t_np = np.matrix.transpose(x_np)

    #print(f"base x:\n{x_np}\ntranspose x:\n{x_t_np}\n")

    MSE_Vals = {}

    #find w vector for given lambda, calc MSE, store in dict with associated lambda to graph
    for lam in range(start_lam, end_lam + 1, 1):
        xT_x = np.matmul(x_t_np, x_np)
        xT_x_lam = xT_x + (lam * np.identity(len(xT_x)))
        x_inv = np.linalg.inv(xT_x_lam)
        xinv_xT = np.matmul(x_inv, x_t_np)
        w_np = np.matmul(xinv_xT, y_np)
        #print(f"ALL THE W'S {lam}:\n{w_np}\n")      #debug
        #print(f"full w:\n{w_np}\n")        #debug
        #w_0_np = w_np[0]                   get w(0)
        #w_np = w_np[1:len(w_np):1]         remove w(0) from w_np
        

        #non linear transform predicted value?????
        
        
        predicted = NonLinearTransformation(x_np, np.matrix(w_np).T, complexity)
        #predicted = np.matmul(x_np, w_np)
        diff = np.subtract(predicted, y_np)
        #print(f"y_np:\n{y_np}")
        #print(f"diff:\n{diff}")

        diff = np.matmul(diff.T, diff)
        #print(f"diff squared:\n{diff}")
        
        MSE = diff / len(x.index)

        MSE_Vals.update({lam : MSE})
        

    return MSE_Vals

def TestLambda(w_train, x_test, y_test, lam, complexity):
    #print(f"Input W:\n{w_train}\nInput X:\n{x_test}\nInput Y:\n{y_test}\n") #debug

    pred_vals = NonLinearTransformation(x_test, w_train, complexity)
    #pred_vals = np.matmul(x_test, np.matrix(w_train).T)
    #print(f"pred vals act\n{pred_vals}\n") #debug
    diff = np.subtract(pred_vals, y_test)
    diff = np.matmul(diff.T, diff)
    #print(f"difference\{diff}\n") #debug
    L2_w = np.matmul(np.matrix(w_train).T, np.matrix(w_train))
    error = diff / len(y_test)
    

    return error


def LambdaError(pred, actual, w, start_lam, end_lam):
    
    MSE_Vals  = {}
    
    diff = np.subtract(pred, actual)
    diff = np.matmul(diff.T, diff)
    err_model = diff / len(w)
    complexity = np.matmul(w.T, w)
    #print(f"err model:\n{err_model}\ncomplexity:\n{complexity}")  #debug

    for lam in range(start_lam, end_lam + 1, 1):
        MSE = err_model + lam * complexity
        MSE_Vals.update({lam : MSE})


    return MSE_Vals


#def CrossValidation



def main():

    #testing-------------------------------------------------------------------------------------------------------------
    #files = {1 : "train-50(1000)-100.csv", 2 : "train-100(1000)-100.csv", 3: "train-150(1000)-100.csv", 4: "train-100-10.csv", 5: "train-100-100.csv",
    #         6: "train-1000-100.csv", 7: "test-100-10.csv", 8: "test-100-100.csv", 9: "test-1000-100.csv"}
    #"C:\Users\donov\Desktop\VS Projects\Python\Data Mining\HW1\HW1\train-50(1000)-100.csv"
    #df_50_1000 = pd.read_excel("train-50(1000)-100.csv")
    #df_100_1000 = pd.read_excel("train-100(1000)-100.csv")
    #df_150_1000 = pd.read_excel("train-150(1000)-100.csv")

    #file selection for testing purposes
    #print(f"{files}")
    ##user input for which file to run
    #num_input = False 
    #while (num_input == False): 
    #    try:
    #        file_select = int(input("Input number of the file to process: "))
    #    except ValueError:
    #        print("Input invalid")

    #    if file_select < 1 or file_select > len(files):
    #        print(f"Input integer 1 through {len(files)}.")
    #    else:
    #        num_input = True

    #df = pd.read_excel(files[file_select])
    #df.Name = files[file_select]
    #testing------------------------------------------------------------------------------------------------------------



    files = {1 : "train-50(1000)-100.csv", 2 : "train-100(1000)-100.csv", 3: "train-150(1000)-100.csv", 4: "train-100-10.csv", 5: "train-100-100.csv",
             6: "train-1000-100.csv", 7: "test-100-10.csv", 8: "test-100-100.csv", 9: "test-1000-100.csv"}

    df_train_50_1000_100 = pd.read_excel("train-50(1000)-100.csv", usecols= lambda c: not c.startswith('Unnamed:'))      #created 1
    df_train_50_1000_100.Name = "train-50(1000)-100"

    df_train_100_1000_100 = pd.read_excel("train-100(1000)-100.csv", usecols= lambda c: not c.startswith('Unnamed:'))    #created 2
    df_train_100_1000_100.Name = "train-100(1000)-100"

    df_train_150_1000_100 = pd.read_excel("train-150(1000)-100.csv", usecols= lambda c: not c.startswith('Unnamed:'))    #created 3
    df_train_150_1000_100.Name = "train-150(1000)-100"


    df_train_100_10 = pd.read_csv("train-100-10.csv", usecols= lambda c: not c.startswith('Unnamed:'))         #train 1
    df_train_100_10.Name = "train-100-10"

    df_train_100_100 = pd.read_csv("train-100-100.csv", usecols= lambda c: not c.startswith('Unnamed:'))       #train 2
    df_train_100_100.Name = "train-100-100"

    df_train_1000_100 = pd.read_csv("train-1000-100.csv", usecols= lambda c: not c.startswith('Unnamed:'))     #train 3
    df_train_1000_100.Name = "train-1000-100"


    df_test_100_10 = pd.read_csv("test-100-10.csv", usecols= lambda c: not c.startswith('Unnamed:'))           #test 1
    df_test_100_10.Name = "test-100-10"

    df_test_100_100 = pd.read_csv("test-100-100.csv", usecols= lambda c: not c.startswith('Unnamed:'))         #test 2
    df_test_100_100.Name = "test-100-100"

    df_test_1000_100 = pd.read_csv("test-1000-100.csv", usecols= lambda c: not c.startswith('Unnamed:'))       #test 3
    df_test_1000_100.Name = "test-1000-100"


    files = [[df_train_100_10, df_test_100_10],          #train 1, test 1
             [df_train_100_100, df_test_100_100],        #train 2, test 2
             [df_train_1000_100, df_test_1000_100],       #train 3, test 3
             [df_train_50_1000_100, df_test_1000_100],    #created 1, test 3
             [df_train_100_1000_100, df_test_1000_100],   #created 2, test 3
             [df_train_150_1000_100, df_test_1000_100]]  #created 3, test 3

    COMPLEXITY_USED = 1



    print(f"Question 2a, training lambda applied to test sets\n---------------------------------------------------------------------------------")
    for i in range(len(files)):
        df_train = files[i][0]
        df_test = files[i][1]

        x_train = GetX(df_train)
        x_0_train = GetXNaught(df_train)
        y_train = GetYVec(df_train)
        #w_0_train = GetW_WithNaught(x_0_train.to_numpy(), y_train.to_numpy())
        #y_pred_train = np.matmul(x_0_train.to_numpy(), w_0_train.to_numpy().T)
        train_errors = MinLambdaValue(x_0_train, y_train, 0, 150, 1)
        print(f"Min error lambda for {df_train.Name}: {min(train_errors, key=train_errors.get)}")
        print(f"Training Error Values:\n{train_errors}\n")

        x_test = GetX(df_test)
        x_0_test = GetXNaught(df_test)
        y_test = GetYVec(df_test)

        #calculate w for a given lambda of the training set, use it to find error in the test set
        Min_MSE = {}
        for lam in range(151):
            x = x_0_train.to_numpy()
            xT_x = np.matmul(x.T, x)
            xT_x_lam = xT_x + (lam * np.identity(len(xT_x)))
            x_inv = np.linalg.inv(xT_x_lam)
            x_inv_xT = np.matmul(x_inv, x.T)
            w_train = np.matmul(x_inv_xT, y_train) #w vector of training set for a given lambda
            #print(f"Train w vector for lam {lam} for {df_train.Name}:\n{train_w}")

            Min_MSE.update({lam : TestLambda(np.matrix(w_train).T, x_0_test.to_numpy(), y_test.to_numpy(), lam, COMPLEXITY_USED)})
            
        print(f"Min error lambda for {df_test.Name}: {min(Min_MSE, key=Min_MSE.get)}")
        print(f"MSE Vlaues:\n{Min_MSE}\n\n\n")
        
        
        plot.scatter(x=train_errors.keys(), y=train_errors.values(), c='b', marker='o', label=df_train.Name)
        plot.scatter(x=Min_MSE.keys(), y=Min_MSE.values(), c='r', marker='s', label=df_test.Name)
        plot.legend(loc='upper left')
        plot.xlabel("Lambda Value")
        plot.ylabel("MSE")
        plot.title(f"Lam/MSE for {df_train.Name} and {df_test.Name}")
        plot.show()

        


    print("---------------------------------------------------------------------------------------------------------------\n\n\n")




  
    print(f"Question 2b, three additional graphs.\n---------------------------------------------------------------------------")
    Question1B = [[df_train_100_100, df_test_100_100],    #created 1, test 3
                  [df_train_50_1000_100, df_test_1000_100],   #created 2, test 3
                  [df_train_100_1000_100, df_test_1000_100]]  #created 3, test 3
    for i in range(len(Question1B)):
        df = Question1B[i][0]
        df_test = Question1B[i][1]

        x_train = GetX(df)
        x_0_train = GetXNaught(df)
        y_train = GetYVec(df)
        w_0_train = GetW_WithNaught(x_0_train.to_numpy(), y_train.to_numpy())
        y_pred_train = np.matmul(x_0_train.to_numpy(), w_0_train.to_numpy().T)
        train_errors = MinLambdaValue(x_0_train, y_train, 1, 150, 1)
        
        x_test = GetX(df_test)
        x_0_test = GetXNaught(df_test)
        y_test = GetYVec(df_test)


        Min_MSE = {}
        for lam in range(1, 151, 1):
            x = x_0_train.to_numpy()
            xT_x = np.matmul(x.T, x)
            xT_x_lam = xT_x + (lam * np.identity(len(xT_x)))
            x_inv = np.linalg.inv(xT_x_lam)
            x_inv_xT = np.matmul(x_inv, x.T)
            w_train = np.matmul(x_inv_xT, y_train) #w vector of training set for a given lambda
            #print(f"Train w vector for lam {lam} for {df_train.Name}:\n{train_w}")
            Min_MSE.update({lam : TestLambda(np.matrix(w_train).T, x_0_test.to_numpy(), y_test.to_numpy(), lam, COMPLEXITY_USED)})


        print(f"Min error lambda for {df.Name}: {min(train_errors, key=train_errors.get)}")
        print(f"MSE Values:\n{train_errors}\n")
        print(f"Min error lambda for {df_test.Name}: {min(Min_MSE, key=Min_MSE.get)}")
        print(f"MSE Values:\n{Min_MSE}\n\n")

        plot.scatter(x=train_errors.keys(), y=train_errors.values(), c='b', marker='o', label=df_train.Name)
        plot.scatter(x=Min_MSE.keys(), y=Min_MSE.values(), c='r', marker='s', label=df_test.Name)
        plot.legend(loc='upper left')
        plot.xlabel("Lambda Value")
        plot.ylabel("MSE")
        plot.title(f"Lam/MSE for {df.Name} and {df_test.Name}")
        plot.show()

    
        
        
    print("---------------------------------------------------------------------------------------------------------------\n\n\n")





    #for each training file, separates data into 10 folds, training on all data except the ith (calculated using start/end point integers which are determined by the iteration and the length of the file)
    #then trains on the ith fold. As each fold is calculated, it is summed in avg_lam, the averaged at the end of the file loop
    print(f"\n\nCross Validation:\n")
    for i in range(len(files)):  #file loop
        df_train = files[i][0]
        df_test = files[i][1]

        min_x_train = GetX(df_train)
        min_x_0_train = GetXNaught(df_train)
        min_y_train = GetYVec(df_train)

        min_x_test = GetX(df_test)
        min_x_0_test = GetXNaught(df_test)
        min_y_test = GetYVec(df_test)


        length = len(df_train.index)
        avg_lam = {k:0 for k in range(151)} #stores lambda value and error of min lam from training applied to test set
        for iteration in range(10):  #fold loop
            start_drop = int((length / 10) * iteration)
            end_drop = int(start_drop + (length / 10))
            
            drop_list = list(range(start_drop, end_drop, 1))
            #print(f"Indexes to drop:{drop_list}\n")

            df_cv_train = df_train.drop(index = drop_list)
            df_cv_test = df_train.iloc[start_drop : end_drop]
            #print(f"\n\nCV Train Frame:\n{df_cv_train}\nCV Test Frame:\n{df_cv_test}\n\n")

            x_train = GetX(df_cv_train)
            x_0_train = GetXNaught(df_cv_train)
            y_train = GetYVec(df_cv_train)
            w_0_train = GetW_WithNaught(x_0_train.to_numpy(), y_train.to_numpy())
            #y_pred_train = np.matmul(x_0_train.to_numpy(), w_0_train.to_numpy().T)

            x_test = GetX(df_cv_test)
            x_0_test = GetXNaught(df_cv_test)
            y_test = GetYVec(df_cv_test)
                
            test_mse = []
            for lam in range(151): #lambda loop
                x = x_0_train.to_numpy()
                xT_x = np.matmul(x.T, x)
                xT_x_lam = xT_x + (lam * np.identity(len(xT_x)))
                x_inv = np.linalg.inv(xT_x_lam)
                x_inv_xT = np.matmul(x_inv, x.T)
                w_train = np.matmul(x_inv_xT, y_train)
                
                test_mse.append(TestLambda(np.matrix(w_train).T, x_0_test.to_numpy(), y_test.to_numpy(), lam, COMPLEXITY_USED))

            #print(f"Test MSE Error for Fold {iteration}:\n{test_mse}") #debug
            
            for key, err in zip(avg_lam, test_mse):
                avg_lam[key] += err

            #print(f"Updated Avg Lambda for Fold {iteration}:\n{avg_lam}\n") #debug

        for key, err in avg_lam.items():
            avg_lam[key] = err/10

        min_lam = min(avg_lam, key=avg_lam.get)

        temp_x = min_x_0_train.to_numpy()
        temp_xT_x = np.matmul(temp_x.T, temp_x) 
        temp_xT_x_lam = temp_xT_x + (min_lam * np.identity(len(temp_xT_x)))
        temp_x_inv = np.linalg.inv(temp_xT_x_lam)
        temp_x_inv_xT = np.matmul(temp_x_inv, temp_x.T)
        temp_w_train = np.matmul(temp_x_inv_xT, min_y_train)
        min_test = {}
        min_test.update({min_lam : TestLambda(np.matrix(temp_w_train).T, min_x_0_test.to_numpy(), min_y_test.to_numpy(), min_lam, COMPLEXITY_USED)})

        print(f"Minimum Cross Validated Lambda for {df_train.Name}: {min_lam}")
        print(f"MSE of Minimum Lambda Applied to Test Set: {min_test[min_lam]}")
        print(f"Averages of Lambda for {df_train.Name}:\n{avg_lam}\n\n")

    return
        


            
           
        
        

        

   
            
             





    #general loop form, testing
    #df = pd.read_excel("train-50(1000)-100.csv")
    ##store all values into separate pandas data frames (full x matrix, x with 1s in x(0), y vector, w vector with and without w(0), and w(0) by itself)
    #x_whole = GetX(df)                               #whole x vector
    #x_whole_0 = GetXNaught(df)                       #whole x vector with x(0) attributes set to 1
    #y_vec = GetYVec(df)                              #observed/target values
    #w_vec_0 = GetW_WithNaught(x_whole_0.to_numpy(), y_vec.to_numpy())      #w vector with w(0)
    #w_0 = w_vec_0.iloc[:, 0]                         #value of w(0)
    #w_vec = w_vec_0.iloc[:, 1:len(w_vec_0.columns)]  #w vector without w(0)
    #y_pred = GetPredictedValues(x_whole, w_vec, w_0) #predicted y values (linear)
    #y_pred_matrix = np.matmul(x_whole_0.to_numpy(), w_vec_0.to_numpy().T) #USE THIS FRO PREDICTED VALEUS
    #MSE = GetMSEMatrix (y_pred_matrix, np.matrix(y_vec).T)
    
   
   

if __name__ == "__main__":
    main()

    #input validation for testing
    #rerun = False 
    #while (rerun == False): 
    #    rerun_input = input("Run new file? (Y/N): ")
    #    rerun_input = rerun_input.lower()
        
    #    if rerun_input not in {'y', 'n'}:
    #        print("Input Y for yes, N for no.")
    #    elif rerun_input == 'y':
    #        rerun = True
    #        main()
    #    else:
    #        rerun = True
