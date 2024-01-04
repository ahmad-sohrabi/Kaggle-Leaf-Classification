# [Kaggle Leaf Classification](https://www.kaggle.com/c/leaf-classification) - Neural Network Approach
In This Project, We have analyzed different hyperparameters of neural networks and preprocessing techniques to see their effect on Accuracy and Loss.
<br />
The whole process is done on [Kaggle Leaf Classification Competition](https://www.kaggle.com/c/leaf-classification).
<br />
For neural network programming, [pytorch](https://pytorch.org/) framework is used. 
<br />

In all plots below, test data means validation set and not kaggle test dataset.
<br />
The First Part See what is the effect of number of layers.
## Number of Neurons Effect
This part consist of 3 types of neural networks all with 2 hidden layers but with 8, 16 & 32 neurons.
<br />
Here is the accuracy plot for 8 neurons network.
<br />
<br />
![8 neurons network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T1-2L-8-Acc.png?raw=true)
<br />
<br />
Here is the accuracy plot for 16 neurons network.
<br />
<br />
![16 neurons network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T1-2L-16-Acc.png?raw=true)

<br />

Here is the accuracy plot for 32 neurons network.
<br />
<br />
![32 neurons network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T1-2L-32-Acc.png?raw=true)

## Number of Layers Effect
This part consist of 3 types of neural networks all with 60 Neurons in the whole network but with 2, 4 & 6 layers.
<br />
Here is the accuracy plot for 2 layers network.
<br />
<br />
![2 layer network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T2-2L-32-Acc.png?raw=true)
<br />
<br />
Here is the accuracy plot for 4 layers network.
<br />
<br />
![4 layer network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T2-4L-32-Acc.png?raw=true)
<br />
Here is the accuracy plot for 6 layers network.
<br />
<br />
![6 layer network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T2-6L-32-Acc.png?raw=true)
<br />

## Learning Rate Effect
This part consist of 3 types of neural networks all with 2 layers and 32 Neurons per hidden layer in the whole network but with 0.01, 0.001 & 0.0001 learning rate.
<br />
Here is the accuracy plot for network with learning rate 0.0001.
<br />
<br />
![0.0001 network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T3-2L-32-Acc-LR%200.0001.png?raw=true)
<br />
<br />
Here is the accuracy plot for network with learning rate 0.001.
<br />
<br />
![0.001 network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T3-2L-32-Acc-LR%200.001.png?raw=true)
<br />

Here is the accuracy plot for network with learning rate 0.01.
<br />
<br />
![0.01 network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T3-2L-32-Acc-LR%200.01.png?raw=true)
<br />

## Activation Function Effect
This part consist of 2 types of neural networks all with 2 layers and 32 Neurons per hidden layer in the whole network but with Tanh & ReLU activation function.
<br />
Here is the accuracy plot for network with Tanh activation function.
<br />
<br />
![Tanh network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T4-2L-32-Acc-Tanh.png?raw=true)
<br />

Here is the accuracy plot for network with ReLU activation function.
<br />
<br />
![ReLU network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/T4-2L-32-Acc-relu.png?raw=true)
<br />

## Selecting Best Hyperparameters with Optimization Method
Using Optuna framework, hyperparameters are tuned. Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.
<br />
Here is the accuracy plot for network with tuned hyperparameters.
<br />
<br />
![tuned network](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/bestResultAcc.png?raw=true)
<br />
## Kaggle Score on Test Dataset
After Hyperparameter tuning in the last part, kaggle score on test data is as below:
<br />
![kaggle test result](https://github.com/ahmad-sohrabi/Kaggle-Leaf-Classification/blob/main/results/kaggle%20result.png?raw=true)
<br />

