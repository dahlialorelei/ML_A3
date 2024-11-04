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
    def __init__(self,learning_rate,epoch,tolerance,C_value,X,Y,X_test,Y_test, do_mini_batch = False, mini_batch_size = 32):

        #initialize the variables
        self.input = X
        self.target = Y
        self.test_input = X_test
        self.test_target = Y_test
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.C = C_value
        self.tol = tolerance
        self.minibatch = do_mini_batch
        self.batch_size = mini_batch_size


        #initialize the weight matrix based on number of features 
        # bias and weights are merged together as one matrix
        # you should try random initialization
     
        #self.weights = np.zeros(X.shape[1])
         # Set the random seed for reproducibility
        seed = 12
        np.random.seed(seed)
        self.weights = np.random.randn(X.shape[1])


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
            hinge_loss = max(0, 1 - y_i * np.dot(x_i, self.weights))
            loss += hinge_loss
        # Normalize hinge loss
        loss_avg = loss / len(X)
        # Add regularization term
        regularization_term = 0.5 * self.C * np.dot(self.weights, self.weights)
        loss_avg += regularization_term
                
        return loss_avg

    def find_smallest_loss(self, test_features, test_output):
        smallest_loss = float('inf')
        best_feature_index = -1

        for i in range(len(test_features)):
            feature = test_features[i:i+1]  # Select the i-th feature vector
            output = test_output[i:i+1]  # Select the corresponding output
            loss = self.compute_loss(feature, output)
            if loss < smallest_loss:
                smallest_loss = loss
                best_feature_index = i

        return best_feature_index, smallest_loss
    
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

            if(self.minibatch):
                # Divide the dataset into mini-batches
                for start in range(0, len(features), self.batch_size):
                    end = start + self.batch_size
                    mini_batch_X = features[start:end]
                    mini_batch_Y = output[start:end]

                    # Compute the gradient for the mini-batch
                    gradient = np.zeros_like(self.weights)
                    for i in range(len(mini_batch_X)):
                        gradient += self.compute_gradient(mini_batch_X[i], mini_batch_Y[i])
                    gradient /= len(mini_batch_X)

                    # Update the weights
                    self.weights = self.weights - (self.learning_rate * gradient)
            else:
                for i, feature in enumerate(features):
                    gradient = self.compute_gradient(feature, output[i])
                    self.weights = self.weights - (self.learning_rate * gradient)

            #print epoch if it is equal to thousand - to minimize number of prints
            if epoch % (self.epoch // 10) == 0 or epoch == self.epoch - 1:
            #if epoch %1 == 0:
                loss = self.compute_loss(features, output)
                val_loss = self.compute_loss(test_features, test_output)
                print("Epoch is: {}, Training Loss is : {}, and Validation Loss is : {}".format(epoch, loss, val_loss))

                # Save loss, validation loss, and epoch values
                loss_values.append(loss)
                val_loss_values.append(val_loss)
                epoch_values.append(epoch)

            
            # Part 1: Early Stopping
            # Compute the loss on the validation set
            current_loss = self.compute_loss(test_features, test_output)

            # Check if the improvement is below the tolerance
            if ((abs(prev_loss - current_loss) < self.tol) and first_time):
                min_epoch = epoch
                early_weights = np.copy(self.weights)
                print("The minimum number of iterations taken are: ", min_epoch)
                first_time = False

            if current_loss < prev_loss:
                best_weights = np.copy(self.weights)

            prev_loss = current_loss

        


            # below code will be required for Part 3
            
            # Part 3
        
        #samples+=1

        print("Training ended...")
        print("Weights are: {}".format(self.weights))

        # Plot loss and validation loss
        plt.plot(epoch_values, loss_values, label='Training Loss')
        plt.plot(epoch_values, val_loss_values, label='Validation Loss')
        plt.axvline(x=min_epoch, color='r', linestyle='--', label='Min Epoch')  # min_epoch line 
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if self.minibatch:
            plt.title('Training and Validation Loss with Mini Batch') 
        else:   
            plt.title('Training and Validation Loss with Batch')  
        #plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.show()

      

        
        return early_weights, best_weights, epoch_values, loss_values, val_loss_values, min_epoch

        # below code will be required for Part 3
        #print("The minimum number of samples used are:",samples)

    def mini_batch_gradient_descent(self, X, Y, test_features, test_output, batch_size):
        #DON'T USE THIS FUNCTION
        self.mini_batch = True
        self.batch_size = batch_size
        early_weights, best_weights = self.stochastic_gradient_descent(X, Y, test_features, test_output)

        return early_weights, best_weights, epoch_values, loss_values, val_loss_values

    def sampling_strategy(self,X,Y):
        x = X[0]
        y = Y[0]
        #implementation of sampling strategy - start

        #Part 3

        #implementation of sampling strategy - start
        return x,y

    def predict(self,X_test,Y_test,weights = None):
        if weights is None:
            weights = self.weights

        # Compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], weights)) for i in range(X_test.shape[0])]
       
        # Compute accuracy
        accuracy = accuracy_score(Y_test, predicted_values)

        # Compute precision
        precision = precision_score(Y_test, predicted_values)

        # Compute recall
        recall = recall_score(Y_test, predicted_values)
        
        return accuracy, precision, recall
        

def part_1(X_train, y_train, X_test, y_test):
    # Model parameters
    C = 1.4
    learning_rate = 0.001
    epoch = 1000
    tol = 1e-5
  
    # Instantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,tolerance=tol,C_value=C,X=X_train,Y=y_train, X_test=X_test, Y_test=y_test)

    # Pre preocess data
    X_norm,Y,X_test_norm = my_svm.pre_process()

    # Train model
    early_weights, best_weights, epoch_values, loss_values, val_loss_values, min_epoch = my_svm.stochastic_gradient_descent(X_norm,Y,X_test_norm,y_test)

    # Compute accuracy
    accuracy, precision, recall = my_svm.predict(X_test_norm, y_test, early_weights)
    print("Accuracy on test dataset from part 1: {}".format(accuracy))
    print("Precision on test dataset from part 1: {}".format(precision))
    print("Recall on test dataset from part 1: {}".format(recall))
    
    return my_svm, loss_values, val_loss_values, min_epoch

def part_2(X_train, y_train, X_test, y_test):
    #model parameters - try different ones
    C = 1.4 
    learning_rate = 0.001 
    epoch = 1000
    batch_size = 32
    tol = 1e-5
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,tolerance=tol,C_value=C,X=X_train,Y=y_train, X_test=X_test, Y_test=y_test, do_mini_batch = True, mini_batch_size = batch_size)

    #pre preocess data
    X_norm,Y,X_test_norm = my_svm.pre_process()

    # train model with mini-batch gradient descent
    #early_weights, best_weights = my_svm.mini_batch_gradient_descent(X_norm,Y,X_test_norm,y_test, batch_size)
    early_weights, best_weights, epoch_values, loss_values, val_loss_values, min_epoch = my_svm.stochastic_gradient_descent(X_norm,Y,X_test_norm,y_test)

    # train model with stochastic gradient descent
    svm, loss_values_stochastic, val_loss_values_stochastic, min_epoch_stochastic = part_1(X_train, y_train, X_test, y_test)

    # compute accuracy
    accuracy, precision, recall = my_svm.predict(X_test_norm, y_test, early_weights)
    print("Accuracy on test dataset from part 2: {}".format(accuracy))
    print("Precision on test dataset from part 2: {}".format(precision))
    print("Recall on test dataset from part 2: {}".format(recall))

    # Plot loss values and validation loss values for both methods
    plt.plot(epoch_values, loss_values, color='b', linestyle='--', label='Mini-Batch Training Loss')
    plt.plot(epoch_values,  val_loss_values, color='orange', linestyle='--', label='Mini-Batch Validation Loss')
    plt.plot(epoch_values, loss_values_stochastic, color='b',  label='Stochastic Training Loss')
    plt.plot(epoch_values,  val_loss_values_stochastic, color='orange', label='Stochastic Validation Loss')
    plt.axvline(x=min_epoch, color='r', linestyle='--', label='Min Epoch for Mini-Batch')  # Vertical line at min_epoch
    plt.axvline(x=min_epoch_stochastic, color='r', label='Min Epoch for Stochastic')  # Vertical line at min_epoch_stochastic
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Mini-Batch and Stochastic Gradient Descent')
    plt.legend()
    plt.show()

    return my_svm

def part_3(X_train, y_train, X_test, y_test):
    #model parameters - try different ones
    C = 1.4 
    learning_rate = 0.001 
    epoch = 1000
    tol = 1e-3
    acc_tol = 0.0001
    prev_accuracy = float('inf')
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,tolerance=tol,C_value=C,X=X_train,Y=y_train, X_test=X_test, Y_test=y_test)

    #pre preocess data
    X_norm,Y,X_test_norm = my_svm.pre_process()
    
    # Small, random subset of samples for training
    size = 0.02 # 5% of the data
    X_initial, X_unlabeled, y_initial, y_unlabeled = train_test_split(X_norm, Y, train_size=size, stratify=Y)

    # Inital training of model
    my_svm.stochastic_gradient_descent(X_initial, y_initial, X_test_norm, y_test)

    # Lists to store accuracy and number of samples
    accuracy_values = []
    num_samples = []

    for _ in range(len(X_unlabeled)):
        # Select the sample with the smallest loss from the unlabeled samples
        best_feat_i, smallest_loss = my_svm.find_smallest_loss(X_unlabeled, y_unlabeled)

        # Add the selected sample to the labeled training set
        X_initial = np.vstack([X_initial, X_unlabeled[best_feat_i]])
        y_initial = np.vstack([y_initial, y_unlabeled[best_feat_i]])

        # Remove the selected sample from the unlabeled set
        X_unlabeled = np.delete(X_unlabeled, best_feat_i, axis=0)
        y_unlabeled = np.delete(y_unlabeled, best_feat_i, axis=0)

        # Retrain the classifier on the updated training set
        my_svm.stochastic_gradient_descent(X_initial, y_initial, X_test_norm, y_test)

        accuracy, precision, recall = my_svm.predict(X_test_norm, y_test)

        # Store accuracy and number of samples
        accuracy_values.append(accuracy)
        num_samples.append(len(X_initial))

        if accuracy > 0.99:
            print("The accuracy is above 99%.")
            break
        if (prev_accuracy - accuracy) < acc_tol:
            print("The accuracy improvement is below the tolerance.")
            break
        
        prev_accuracy = accuracy
    
    # Plot accuracy values vs the number of samples 
    plt.plot(num_samples, accuracy_values, marker='o')
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Samples Used for Training')
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.xticks(np.arange(min(num_samples), max(num_samples) + 1, 1))
    plt.show()

    accuracy, precision, recall = my_svm.predict(X_test_norm, y_test)
    print("Accuracy on test dataset from part 3: {}".format(accuracy))
    print("Precision on test dataset from part 3: {}".format(precision))
    print("Recall on test dataset from part 3: {}".format(recall))
    print("The number of samples used:",len(X_initial))

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
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)



#my_svm = part_1(X_train,y_train,X_test,y_test)
#my_svm = part_2(X_train,y_train,X_test,y_test)
my_svm = part_3(X_train,y_train,X_test,y_test)

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
