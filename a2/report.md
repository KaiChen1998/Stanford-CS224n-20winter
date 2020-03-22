# Assignment 2: Written Part

- (a): cross entropy for one hot ground truth
  $$
  \begin{aligned}
      - \sum_{w\in Vocab}y_w\log(\hat{y}_w) &= - [y_1\log(\hat{y}_1) + \cdots + y_o\log(\hat{y}_o) + \cdots + y_w\log(\hat{y}_w)] \\
      & = - y_o\log(\hat{y}_o) \\
      & = -\log(\hat{y}_o) \\
      & = -\log \mathrm{P}(O = o | C = c)
  \end{aligned}
  $$
  

- (b): gradients for center word vector
  $$
  \begin{aligned}
  \frac{\partial J}{\partial v_c} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial v_c} \\
  &= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial v_c} \\
  &= U^T(\hat{y} - y)^T
  \end{aligned}
  $$

- (c): gradients for outside word vector

  Using naive softmax won't cause sparse optimization problem.
  $$
  \begin{aligned}\frac{\partial J}{\partial U} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial U} \\&= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial U} \\&= v_c(\hat{y} - y)^T\end{aligned}
  $$

- (d): gradients for sigmoid
  $$
  \begin{aligned}
  \sigma^\prime(x)=\sigma(x)(1-\sigma(x))
  \end{aligned}
  $$

- (e): gradients for negative sampling
  $$
  \begin{aligned}
  \frac{\partial J}{\partial u_o} &= -(1-\sigma(u_0^Tv_c))v_c \\
  \frac{\partial J}{\partial u_k} &= (1-\sigma(-u_k^Tv_c))v_c \\
  \frac{\partial J}{\partial v_c} &= -(1-\sigma(u_0^Tv_c))u_0+\Sigma_k(1-\sigma(-u_k^Tv_c))u_k
  \end{aligned}
  $$

- (f): total gradients: just sum the single gradients up

