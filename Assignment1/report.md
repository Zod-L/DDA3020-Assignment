# Report-Assignment1
## Linear Model
In this assignment, I use ridge linear regression to fit the data. Specifically, given an input selected features $x\in R^d$, where d is the number of features, the predicted housing price $\hat{y}$ can be predicted by
$$
\hat{y} = \bar{x}^T\bar{w}
$$ 
where
$$
\bar{x} = [1, x_1, x_2, ..., x_d]^T \in R^{d+1}
$$ 


$$
\bar{w} = [b, w_1, w_2, ..., w_d]^T \in R^{d+1}
$$ 


## Loss Function
Since I use ridge linear regression, the parameters $\bar{w}$ is trained by minimizing the loss function:
$$
J(w) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\bar{w}}(x_i) - y_i)^2 + \frac{\lambda}{2}\bar{w}^T\bar{w}
$$
where m is the number of training samples.
The loss function can also be written as
$$
J(w) = \frac{1}{2m}||X\bar{w} - Y||^2 + \frac{\lambda}{2}\hat{w}^T\hat{w}
$$
where
$$
X = 
\begin{matrix} 
\bar{x}_1^T \\ 
\bar{x}_2^T \\ 
...         \\
\bar{x}_m^T 
\end{matrix}
\in R^{m\times(d+1)}
$$


$$
Y = 
\begin{matrix} 
y_1 \\ 
y_2  \\ 
...  \\
y_m
\end{matrix}
\in R^{m\times(d+1)}
$$




$$
\hat{w} = [0, w_1, w_2, ..., w_d]^T \in R^{d+1}
$$



## Gradient Descend
The procedure of gradient descend can be summarized as:
* Calculate the gradient of loss function w.r.t. weight 
$$
\nabla{J(\bar{w})} = \frac{\partial J}{\partial\bar{w}}
$$

* Calculate learning rate $\alpha$ by Armijo Line Search
* Update w by $\bar{w}=\bar{w}-\alpha\nabla{J(\bar{w})}$
* Check whether the procedure should be terminated
* Go back to step 1 if not terminated 
$$
$$
The gradient can be calculated as:
$$
\nabla{J(\bar{w})} = \frac{1}{m}X^T(X\bar{w} - Y) + \lambda\hat{w}
$$
The learning rate $\alpha$ for each step is selected from a set of weights $\{1, \sigma, \sigma^2, \sigma^3, ....\}$ by tring through the set starting from 1, until 
$$
J(w^k-\alpha_k\nabla{J(w^k)}) - J(w^k) \leq -\gamma\alpha_k\nabla{J(w^k)}^T\nabla{J(w^k)}
$$

Whether to terminate the procedure is determined by the loss difference before and after the weight is updated. The procedure is terminated when
$$
||J(w^k) - J(w^k-\alpha_k\nabla{J(w^k)})||^2 \leq threshold
$$
Therefore, in total, there are 4 hyperparamters: $\lambda$, $\sigma$, $\gamma$ and $threshold$.


## Features Selection
In order to avoid overfitting, instead of simly using all features, I only select several features for price prediction. In this report, only part of analysis plots will be shown. To see all plots, please refer to code.ipynb. Features that has an obvious linear relaionship with the price will be chosen. Specifically, features rm, age, dis and lstat are chosen. As is shown in the following plot, these features have strong relationship with price.

<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/age.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 1: Relationship between age and medv</div>
</div><br><br>

<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/dis.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 2: Relationship between dis and medv</div>
</div><br><br>


<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/lstat.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 3: Relationship between lstat and medv</div>
</div><br><br>



<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/lstat.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 4: Relationship between rm and medv</div>
</div><br><br>
As is shown in the heatmap, these features are not depenednt on each other. None of the two features are of the same distribution.


<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/heatmap.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 5:Heatmap</div>
</div><br><br>


However, there exists a weak correlation between some pairs, as is shown in relevance plot. Here only several of these plots will be shown. To see all plots, please refer to code.ipynb.


<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/age-dis.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 6:Relation between age and dis</div>
</div><br><br>


<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/rm-lstat.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 7:Relation between rm and lstat</div>
</div><br><br>


## Training and Testing
10 experiments are conducted. In each experiment, the data is randomly split into training and testing set in a ratio of 8:2. In each experiment, the linear regression model is trained from random initialization, and the tested by the testing set.
### Training
In training, the hyperparameters are set as
$$
\lambda=0.01
$$


$$
\sigma=0.5
$$

$$
\gamma=0.1
$$
$$
threshold=10^{-6}
$$
In 10 experiments, the training is terminated at around 170 to 200 iterations. Two plots of loss function value over iterations are shown here. To see all plots, please refer to code.ipynb.


<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/exp1.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 8:Loss over iteration at experiment1</div>
</div><br><br>

<div align=center>
<img src=/home/liyi/DDA3020-Assignment/Assignment1/im/exp10.png width="80%">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
     display: inline-block; color: #999; padding: 2px;">Figure 9:Loss over iteration at experiment10</div>
</div><br><br>


### Testing
RMSE is used in testing the model.
$$
rmse(w) = \sqrt{\frac{1}{k}\sum_{i=1}^{k}(f_{\bar{w}}(x_i) - y_i)^2}
$$
where k is the number of testing sample. The mean testing rmse over 10 experiments is 3.807362106194172. To see the testing results of all 10 experiments, please refer to code.ipynb.