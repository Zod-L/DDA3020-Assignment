# Report-Assignment2
## Overview
In this assignment, I implement two classification algorithms:
- Decision Tree
- Multilayer Perceptron(MLP) neural network 

using sklearn. Penguins classification and Fashion-MNIST image classification tasks are solved.

## Penguins Classification With Decision Tree
### Data Cleaning and Analysis
In this task, penguins are classified to different type. I try different decision-tree-based algorithms for this task. I explore basis decision tree, bagging of tree and random forest algorithms. Before training the model, data is first cleaned. Those data with incomplete features are removed from the training and testing set. After cleaning the data, 333 of the 344 samples are left. These data is then splitted into training and testing set. 75% of the 333 samples are used for training while the rest 25% are used for testing. The relation between penguin types and different attributes are shown in the following histogram
<div align=center>
<img src=im/island.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 1: Histogram of number of penguins of different type with different islands</div>
</div><br><br>



<div align=center>
<img src=im/bill_length.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 2: Histogram of number of penguins of different type with different bill length, in mm</div>
</div><br><br>



<div align=center>
<img src=im/bill_depth.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 3: Histogram of number of penguins of different type with different bill depth, in mm</div>
</div><br><br>


<div align=center>
<img src=im/flipper.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 4: Histogram of number of penguins of different type with different flipper length, in mm</div>
</div><br><br>


<div align=center>
<img src=im/mass.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 5: Histogram of number of penguins of different type with different body mass, in g</div>
</div><br><br>


<div align=center>
<img src=im/sex.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 6: Histogram of number of penguins of different type with different sex</div>
</div><br><br>

From the above histogram, it can be preliminary observed that bill, flliper and body mass information may be more important than island and sex information.

### Basis Decision Tree
Basis decision tree algorithm is applied to the task, with only one tree bulit. Different maximum depth and maximum leaf size setting are used, which alongs with their performace are summarize in the following table
|  depth   |   leaf size   |Train error|Test error|
|----------|---------------|-----------|----------|
|    3     |       1       |   0.020   |  0.036   |
|    3     |       5       |   0.040   |  0.024   |
|    3     |      25       |   0.056   |  0.036   |
|    5     |       1       |   0.000   |  0.012   |
|    5     |       5       |   0.040   |  0.024   |
|    5     |      25       |   0.056   |  0.036   |
| no limit |       1       |   0.000   |  0.012   |
| no limit |       5       |   0.040   |  0.024   |
| no limit |      25       |   0.056   |  0.036   |

The tree structures are shown in figure 7 to figure 15.
<div align=center>
<img src=im/31.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 7: Tree structure of max depth 3, leaf size 1</div>
</div><br><br>



<div align=center>
<img src=im/35.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 8: Tree structure of max depth 3, leaf size 5</div>
</div><br><br>


<div align=center>
<img src=im/325.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 9: Tree structure of max depth 3, leaf size 25</div>
</div><br><br>


<div align=center>
<img src=im/51.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 10: Tree structure of max depth 5, leaf size 1</div>
</div><br><br>


<div align=center>
<img src=im/55.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 11: Tree structure of max depth 5, leaf size 5</div>
</div><br><br>


<div align=center>
<img src=im/525.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 12: Tree structure of max depth 5, leaf size 25</div>
</div><br><br>



<div align=center>
<img src=im/no1.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 13: Tree structure without limiting depth, leaf size 1</div>
</div><br><br>


<div align=center>
<img src=im/no5.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 14: Tree structure without limiting depth, leaf size 5</div>
</div><br><br>


<div align=center>
<img src=im/no25.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 15: Tree structure without limiting depth, leaf size 25</div>
</div><br><br>

As is shown in the above figure, bill, flliper and body mass information are frequently used, while sex and island information are seldom appear as a split criteria, which is aligned with the observation on data histogram.

### Bagging of Trees
To mitigate overfitting, bagging can be used. The basic idea is to build multiple trees and make the final decision based on output from all trees. Different maximum depth and number of trees setting are explored, which alongs with their performace are summarize in the following table


|  depth   |   # of tree   |Train error|Test error|
|----------|---------------|-----------|----------|
|    3     |       5       |   0.016   |  0.036   |
|    5     |       5       |   0.000   |  0.000   |
| no limit |       5       |   0.008   |  0.024   |
|    3     |      10       |   0.024   |  0.024   |
|    5     |      10       |   0.000   |  0.012   |
| no limit |      10       |   0.004   |  0.012   |
|    3     |      50       |   0.012   |  0.024   |
|    5     |      50       |   0.000   |  0.000   |
| no limit |      50       |   0.000   |  0.000   |


### Random Forests
Compared with Bagging of Trees Algorithm, in Random Forests, not only multiple trees will be trained, but also each tree will only utilize a subset of attributes. Different tree number, number of candidate attributes to split in every step setting are explored, which alongs with their performace are summarize in the following table

|    m     |   # of tree   |Train error|Test error|
|----------|---------------|-----------|----------|
|    1     |       5       |   0.076   |  0.214   |
|    3     |       5       |   0.000   |  0.071   |
|    6     |       5       |   0.004   |  0.000   |
|    1     |      10       |   0.100   |  0.238   |
|    3     |      10       |   0.004   |  0.000   |
|    6     |      10       |   0.000   |  0.000   |
|    1     |      50       |   0.044   |  0.214   |
|    3     |      50       |   0.000   |  0.000   |
|    6     |      50       |   0.000   |  0.000   |

When the number of available attribute is very small, both training and testing has a bad performance. When the number of available attribute is medium, its training error is a little bit higher than using the full set of attributes, but the tesing error is close, since it has both smaller bias and variance. As is shown in figure 16 and 17, as the number of tree increase, bias has a sharp impulse and then back to normal, while variance is decreasing.

<div align=center>
<img src=im/bias.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 16: Change of bias with increasing number of tree</div>
</div><br><br>


<div align=center>
<img src=im/variance.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 17: Change of variance with increasing number of tree</div>
</div><br><br>

## Fashion-MNIST Recognition Using MLP
In this task, I use MLP to recognize the image in Fashion-MNIST. Specifically, given an image with a resolution of 28x28, the network extract its features through hidden layers, and finally output the confidence of each class. Different number of hidden layers, hidden nodes of each hidden layers and optimizer are experienced. Besides, other hyperparameters are in the default setting.



|hyperparameters|   Value   |Explanation|
|---------------|-----------|-----------|
|hidden_layer_sizes|50/200/784 * 1/2/3|How many neurons in each hidden layer|
|activation|relu|The activation function after the neurons of each hidden layer|
|solver|sgd/adam|The optimizer used to update the weights of neurons. Different optimizer has different optimizing strategy|
|batch_size|200|Number of samples within a batch. The optimizer will only update weights after finishing calculating a complete batch|
|learning_rate|0.001,fixed|The weights multiplying to the gradient for each step of updating|
|max_iter|200|The maximum number of epochs. Going through the training data once is counted as 1 epoch|
|tol|1e-4|If the difference before and after an update is less than this value, training stop|


As is shown in the following table, different number of parameters and optimizer can have great effect on the training and testing error. With the same optimizer, as the number of hidden nodes or the number of hidden layers increases, the training error has an obvious drop. However, the test error does not drop so much. The reason is that as the number of parameters increase, the network tend to overfit the training set.
|    # of hidden layers   |    # of hidden nodes   |  optimizer  |Train error |Test error|
|-------------------------|------------------------|-------------|------------|----------|
|            1            |           50           |     SGD     |    0.402   |  0.414   |
|            1            |           50           |     Adam    |    0.114   |  0.149   |
|            1            |           200          |     SGD     |    0.113   |  0.149   |
|            1            |           200          |     Adam    |    0.084   |  0.130   |
|            1            |           784          |     SGD     |    0.060   |  0.129   |
|            1            |           784          |     Adam    |    0.116   |  0.138   |
|            2            |           50           |     SGD     |    0.800   |  0.800   |
|            2            |           50           |     Adam    |    0.080   |  0.129   |
|            2            |           200          |     SGD     |    0.900   |  0.900   |
|            2            |           200          |     Adam    |    0.054   |  0.113   |
|            2            |           784          |     SGD     |    0.900   |  0.900   |
|            2            |           784          |     Adam    |    0.058   |  0.114   |
|            3            |           50           |     SGD     |    0.900   |  0.900   |
|            3            |           50           |     Adam    |    0.074   |  0.131   |
|            3            |           200          |     SGD     |    0.094   |  0.139   |
|            3            |           200          |     Adam    |    0.038   |  0.108   |
|            3            |           784          |     SGD     |    0.028   |  0.125   |
|            3            |           784          |     Adam    |    0.027   |  0.107   | 

For different optimizers, as is shown from figure 16 to figure 33, Adam optimizer always peform better than SGD optimizer. Adam converge much faster than SGD in model with a large parameters set. This is because Adam adapt learning rate based on the average of the second moments of the gradients rate on each step, while SGD does not adjust the learning rate. For this reason, in large model, SGD often fails to reach the global optimum, but stuck at a point, while Adam can always reach the optimum. 

<div align=center>
<img src=im/sgd1x50.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 18: MLP with 1 hidden layers, each with 50 neurons, optimized by sgd optimizer </div>
</div><br><br>

<div align=center>
<img src=im/adam1x50.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 19: MLP with 1 hidden layers, each with 50 neurons, optimized by adam optimizer </div>
</div><br><br>


<div align=center>
<img src=im/sgd1x200.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 20: MLP with 1 hidden layers, each with 200 neurons, optimized by sgd optimizer </div>
</div><br><br>


<div align=center>
<img src=im/adam1x200.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 21: MLP with 1 hidden layers, each with 200 neurons, optimized by adam optimizer </div>
</div><br><br>


<div align=center>
<img src=im/sgd1x784.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 22: MLP with 1 hidden layers, each with 784 neurons, optimized by sgd optimizer </div>
</div><br><br>


<div align=center>
<img src=im/adam1x784.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 23: MLP with 1 hidden layers, each with 784 neurons, optimized by adam optimizer </div>
</div><br><br>


<div align=center>
<img src=im/sgd2x50.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 24: MLP with 2 hidden layers, each with 50 neurons, optimized by sgd optimizer </div>
</div><br><br>


<div align=center>
<img src=im/adam2x50.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 25: MLP with 2 hidden layers, each with 50 neurons, optimized by adam optimizer </div>
</div><br><br>



<div align=center>
<img src=im/sgd2x200.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 26: MLP with 2 hidden layers, each with 200 neurons, optimized by sgd optimizer </div>
</div><br><br>


<div align=center>
<img src=im/adam2x200.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 27: MLP with 2 hidden layers, each with 200 neurons, optimized by adam optimizer </div>
</div><br><br>



<div align=center>
<img src=im/sgd2x784.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 28: MLP with 2 hidden layers, each with 784 neurons, optimized by sgd optimizer </div>
</div><br><br>


<div align=center>
<img src=im/adam2x784.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 29: MLP with 2 hidden layers, each with 784 neurons, optimized by adam optimizer </div>
</div><br><br>



<div align=center>
<img src=im/sgd3x50.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 30: MLP with 3 hidden layers, each with 50 neurons, optimized by sgd optimizer </div>
</div><br><br>



<div align=center>
<img src=im/adam3x50.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 31: MLP with 3 hidden layers, each with 50 neurons, optimized by adam optimizer </div>
</div><br><br>



<div align=center>
<img src=im/sgd3x200.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 32: MLP with 3 hidden layers, each with 200 neurons, optimized by sgd optimizer </div>
</div><br><br>


<div align=center>
<img src=im/adam3x200.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 33: MLP with 3 hidden layers, each with 200 neurons, optimized by adam optimizer </div>
</div><br><br>


<div align=center>
<img src=im/sgd3x784.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 34: MLP with 3 hidden layers, each with 784 neurons, optimized by sgd optimizer </div>
</div><br><br>


<div align=center>
<img src=im/adam3x784.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 35: MLP with 3 hidden layers, each with 784 neurons, optimized by adam optimizer </div>
</div><br><br>