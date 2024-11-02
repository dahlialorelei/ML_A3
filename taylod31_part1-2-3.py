
# Author: Swati Mishra
# Created: Sep 23, 2024
# License: MIT License
# Purpose: This python file includes boilerplate code for Assignment 3

# Usage: python support_vector_machine.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added boilerplate code

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #CHECK THIS IS ALLOWED

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

class svm_():
    def __init__(self,learning_rate,epoch,tolerance,C_value,X,Y,X_test,Y_test,sigma=1.0):

        #initialize the variables
        self.input = X
        self.target = Y
        self.test_input = X_test
        self.test_target = Y_test
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.C = C_value
        self.tol = tolerance
        self.sigma = sigma

        #initialize the weight matrix based on number of features 
        # bias and weights are merged together as one matrix
        # you should try random initialization
     
        #self.weights = np.zeros(X.shape[1])
         # Set the random seed for reproducibility
        seed = 12
        np.random.seed(seed)
        self.weights = np.random.randn(X.shape[1])

    def rbf_kernel(self, x1, x2):
        # Compute the squared Euclidean distance
        distance = np.sum((x1 - x2) ** 2)
        # Compute the RBF kernel
        return np.exp(-distance / (2 * self.sigma ** 2))

    def pre_process(self,):

        #using StandardScaler to normalize the input
        
        scalar = StandardScaler().fit(self.input)
        X_ = scalar.transform(self.input)

        Y_ = self.target 

        scalar = StandardScaler().fit(self.test_input)
        X_test_ = scalar.transform(self.test_input)
        
        '''
        scaler = StandardScaler()
        X_ = scaler.fit_transform(self.input)
        X_test_ = scaler.transform(self.test_input)
        Y_ = self.target 
        '''

        return X_,Y_,X_test_ 
    
    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    def compute_loss(self,X,Y):
        # calculate hinge loss
        loss = 0
        for i in range(len(X)):
            x_i = X[i]
            y_i = Y[i]
            #K = np.array([self.rbf_kernel(x_i, x_j) for x_j in X])
            #hinge_loss = max(0, 1 - y_i * np.dot(K, self.weights))
            hinge_loss = max(0, 1 - y_i * np.dot(x_i, self.weights))
            loss += hinge_loss
        # Normalize hinge loss
        loss_avg = loss / len(X)
        # Add regularization term
        regularization_term = 0.5 * self.C * np.dot(self.weights, self.weights)
        loss_avg += regularization_term
                
        return loss_avg
    
    def stochastic_gradient_descent(self,X,Y,test_features,test_output):
        print("Training started...")
        prev_loss = float('inf')
        first_time = True
        min_epoch = 0

        samples = 0

        # Lists to store loss and validation loss values
        loss_values = []
        val_loss_values = []
        epoch_values = []
        early_weights = np.copy(self.weights)

        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

            #print epoch if it is equal to thousand - to minimize number of prints
            if epoch % (self.epoch // 10) == 0:
            #if epoch %1 == 0:
                loss = self.compute_loss(features, output)
                val_loss = self.compute_loss(test_features, test_output)
                print("Epoch is: {}, Training Loss is : {}, and Validation Loss is : {}".format(epoch, loss, val_loss))

                # Save loss, validation loss, and epoch values
                loss_values.append(loss)
                val_loss_values.append(val_loss)
                epoch_values.append(epoch)

            # Check for convergence
            # Compute the loss on the validation set
            current_loss = self.compute_loss(test_features, test_output)

            # Check if the improvement is below the tolerance
            if (abs(prev_loss - current_loss) < self.tol and first_time):
                min_epoch = epoch
                early_weights = np.copy(self.weights)
                print("The minimum number of iterations taken are: ", min_epoch)

                first_time = False

            prev_loss = current_loss
            
            # Part 1

        

            #check for convergence - end

            # below code will be required for Part 3
            
            # Part 3
        
        #samples+=1

        print("Training ended...")
        print("weights are: {}".format(self.weights))

        # Plot loss and validation loss
        plt.plot(epoch_values, loss_values, label='Training Loss')
        plt.plot(epoch_values, val_loss_values, label='Validation Loss')
        plt.axvline(x=min_epoch, color='r', linestyle='--', label='Min Epoch')  # Add vertical line at min_epoch
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.show()

        return early_weights

        # below code will be required for Part 3
        #print("The minimum number of samples used are:",samples)

    def mini_batch_gradient_descent(self,X,Y,batch_size):

        # mini batch gradient decent implementation - start

        # Part 2

        # mini batch gradient decent implementation - end

        print("Training ended...")
        print("weights are: {}".format(self.weights))

    def sampling_strategy(self,X,Y):
        x = X[0]
        y = Y[0]
        #implementation of sampling strategy - start

        #Part 3

        #implementation of sampling strategy - start
        return x,y

    def predict(self,X_test,Y_test,weights):

        '''
        # Compute predictions on test set
        predicted_values = []
        for i in range(X_test.shape[0]):
            # Compute the RBF kernel values for the i-th test sample against all training samples
            K = np.array([self.rbf_kernel(X_test[i], x_j) for x_j in X_test])
            # Compute the dot product with the weights
            prediction = np.sign(np.dot(K, self.weights))
            predicted_values.append(prediction)
        '''


        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], weights)) for i in range(X_test.shape[0])]
        #predicted_values = [np.sign(np.dot([self.rbf_kernel(X_test[i], x_j) for x_j in self.input], self.weights)) for i in range(X_test.shape[0])]
        #compute accuracy
        accuracy = accuracy_score(Y_test, predicted_values)
        #print("Accuracy on test dataset: {}".format(accuracy))

        #compute precision - start
        # Part 2
        #compute precision - end

        #compute recall - start
        # Part 2
        #compute recall - end
        return accuracy
        #return np.array(predicted_values)

def part_1(X_train, y_train, X_test, y_test):
    #model parameters - try different ones
    C = 1.4
    learning_rate = 0.001
    epoch = 1000
    tol = 1e-5
  
    #instantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,tolerance=tol,C_value=C,X=X_train,Y=y_train, X_test=X_test, Y_test=y_test)

    #pre preocess data
    X_norm,Y,X_test_norm = my_svm.pre_process()

    # train model
    weights = my_svm.stochastic_gradient_descent(X_norm,Y,X_test_norm,y_test)

    # compute accuracy
    accuracy = my_svm.predict(X_test_norm,y_test, weights)
    print("Accuracy on test dataset from part 1: {}".format(accuracy))

    return my_svm

def part_2(X_train,y_train):
    #model parameters - try different ones
    C = 0.001 
    learning_rate = 0.001 
    epoch = 5000
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data

    # select samples for training

    # train model


    return my_svm

def part_3(X_train,y_train):
    #model parameters - try different ones
    C = 0.001 
    learning_rate = 0.001 
    epoch = 5000
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data

    # select samples for training

    # train model


    return my_svm

#Load datapoints in a pandas dataframe
print("Loading dataset...")
data = pd.read_csv('data1.csv')

# drop first and last column 
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

#segregate inputs and targets

#inputs
X = data.iloc[:, 1:]

#add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()

#converting categorical variables to integers 
# - this is same as using one hot encoding from sklearn
#benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
#transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

# split data into train and test set using sklearn feature set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)



my_svm = part_1(X_train,y_train,X_test,y_test)

'''
my_svm = part_2()

my_svm = part_3()

#normalize the test set separately
scalar = StandardScaler().fit(X_test)
X_Test_Norm = scalar.transform(X_test)
# testing the model
print("Testing model accuracy...")
my_svm.predict(X_Test_Norm,y_test)
'''
