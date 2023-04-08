# Report-Assignment2
## Overview
In this assignment, I solve the wine analysis problem. Specifically, given its 13 features, a wine should be classified one of the three classes. To conduct such a multi-classes classification task, support verctor machine(SVM) is utilized.
## Support Vector Machine
### Hard Margin SVM
The basic idea of SVM is to maximize the minimum distance between data and the decision boundary(the so-called margin). Hence, the optimization problem can be formulated as:
$$
max_{w,b}min_{i}\frac{y_i(w^Tx_i+b)}{||w||}
$$
where w, b are the parameter of the linear classifier, and $(x_i, y_i)$ are the supervised training data pair. <br>
Since scaling (w,b) by some factor does not affect the objective function, the solution space of (w,b) can be reduced to a subspace where 
$$
y_i(w^T \hat{x}+b)=1
$$
where $\hat{x}$ is the data sample that is closest to the decision boundary. Hence the optimization problem can be converted to
$$
min_{w,b} \frac{1}{2}||w||^2 \\
s.t. \ \ y_i(w^Tx_i+b)\geq1,\forall i
$$
### SVM With Slack Variables
For hard margin SVM, it is supposed that the data is linear separable. However, in real case, this is often impossible. Hence, slack variables are needed to relax this constraint. Specifically, each data sample is allowed to cross the margin even the decision boundary, but the number of these samples should not be too high. The problem can be formulated as
$$
min_{w,b} \frac{1}{2}||w||^2 + C\sum^m_i\xi_i \\
s.t. \ \ y_i(w^Tx_i+b)\geq1-\xi_i ,\forall i \\
\xi \geq 0
$$


### Solving SVM With Dual
By using KKT conditions, the dual problem of the original optimization of SVM with slack variable can be converted to
$$
max_{\alpha}\sum^m_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j \\
s.t. \sum^m_i\alpha_iy_i=0 \\
0 \leq \alpha_i \leq C \\
$$
w can be obtained by
$$
w=\sum_i^m\alpha_iy_ix_i
$$
b can be obtained by
$$
b=\frac{1}{|M|}\sum_{j\in M}(y_j-\sum_i^m\alpha_iy_ix_i^Tx_j)
$$

### SVM With Kernel
Since in many cases, the training samples are not linearly separable, it is necessary to project data into a dimension that can be linearly splitted. Hence kernel is needed. the simple dot product $x_ix_j$ in the objective function is replaced by more complicated and flexible kernels $k(x_i, x_j)$. In this task, different kernels are tested.


### Training and Testing Setting
In this assignment, I use 80% of the data as training set, with a total number of 142, and the rest 20% of the data is used for testing, with 36 samples.

### Linear SVM Without Slack Variables
In this task, the kernel of SVM is the simplest linear model, and no slack variables are used. The training error rate is 0, meaning that the model well split the training data without error. However the testing error rate is 0.0278. Some samples are misclassified due to overfitting. For more information such as the support vector, weight and bias, please refer to A2_119010156.ipynb.

### Linear SVM With Slack Variables

| C |Train error|Test error|
|---|-----------|----------|
|0.1|  0.0141   |  0.0278  |
|0.2|  0.0211   |  0.0278  |
|0.3|  0.0211   |  0.0000  |
|0.3|  0.0211   |  0.0000  |
|0.5|  0.0211   |  0.0000  |
|0.6|  0.0211   |  0.0000  |
|0.7|  0.0211   |  0.0000  |
|0.8|  0.0070   |  0.0000  |
|0.9|  0.0211   |  0.0000  |
|1.0|  0.0211   |  0.0000  |
As is shown in the table, when the constraint put on the slack variables are small, the classifer tend to output a worse test error, meaning an underfit. As the constraint on slack variables grow stronger, the test error decrease and eventually fall to 0, which is suitable. For more information please refer to A2_119010156.ipynb.


### SVM With Kernels
| kernel  |Train error|Test error|
|---------|-----------|----------|
| 2nd-poly|  0.0000   |  0.0278  |
| 3nd-poly|  0.0000   |  0.0278  |
|   RBF   |  0.0070   |  0.0000  |
|Sigmoidal|  0.5634   |  0.4444  |
From the table, it can be concluded that RBF is the most suitable kernel, while Sigmoidal fail to map the data space into a linear separable space.
