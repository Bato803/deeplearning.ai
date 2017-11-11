# Course Note 

## Regularization (R)

1. Regularization of Logistic Regression: 
  - Often regularize w instead of b, w is a vector while b is a scalar. Most parameters are in w. 
  - If we use L1 norm regularization, w would become more sparse. (May help compressing the model, but the difference is not obvious)
  - L2 norm regularization is much more often. 

2. Regularization on NN:
  - Frobenuius norm: sum of the square of elements in weight matrix. 
  - With L2-norm, Weight matrix is becoming smaller, L2-norm is also called "weight decay"
  
3. Why R reduce overfitting?
  - Less weight -> 'zero' out the impact of hidden units -> simpler network. 
  - tanh activation function -> (less weight)-(smaller z) -> linear region of tanh -> every layer would be roughly linear. 
