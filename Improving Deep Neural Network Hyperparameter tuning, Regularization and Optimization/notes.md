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

4. Dropout
    - Inverted Dropout(Remember to divide keep_prob to ensure the expected value to be the same)
    - Keep_prob can be different for different layer
    - Intuition1: Knocking out neurons -> smaller network -> regularing effect
    - Intuition2: Each neuron can't rely on only one feature, and much more motivated to spread out weight -> shrinking weights.
    - Dropout doesn't always generalize to other discipline. Although in CV. it's almost default. Remember, Dropout is for regularization!
    - Dropout makes cost function less well defined!
    
            When implement dropout for backpro, remember to scale the derivative by keep_pro, just as you did in forward-pro for the activation values. 
    
 5. Other techniques
    - Data Augmentation. (Inexpensive way)
    - Early Stopping (According to the error rate of dev set)
    
## Optimization

1. Normalizing input features -> affect your cost function shape(more round and easier to optimize) -> speed up gradient descent.

2. If the elements in weight matrix is a bit larger than one, in a very deep network, the activation might explode. Conversely, if they are less than one, the activation might vanish. Same with the gradients in BP. 

3. Partial solution for vanishing/exploding gradient problem: Careful choice of initilization. Concretely, set the input feature for each layer to be mean zero and standard variance. Intuition behind this: Hope all the weights matrix not too much bigger/smaller than 1. 

        For Relu: np.random.randn(shape) * np.sqrt(2/n^{l-1}) . (Setting the variance to be 2/n) 

        For tanh: tanh, the last term becomes np.sqrt(1/n^{l-1}) (Xavier initilization)
        
        Alternative: np.sqrt(2/(n^{l-1}+n^{l}))
 4. When we do the numeric difference gradient checking, if epsilon is the order of 10^{-7}, then if the difference between the numeric and BP is the order of 10^{-7}, that's good. If it's the order of 10^{-3}, that's bad!!
  
 5. Implement Gradient Checking without Dropout!
  
 6. Train the network for some time so that w, b can wander away from zero. And then do gradient checking. 
 
 ## Optimization Algorithms
 
 1. Using SGD might cause you losing the speedup from vectorization. 
 
 2. Choosing Mini-batch size rule:
    1. If small training set(<2000), just use batch GD.
    2. Typical mini batch size: 64, 128, 256, 512.
    3. Make sure mini-batch size fits into your CPU or GPU.
 
 3. Exponentially weighted Averages:
    - v0 = 0
    - v1 = 0.9 * v0 + 0.1 * a1
    - v2 = 0.9 * v1 + 0.1 * a2
    - ...
    - vt = 0.9 * v(t-1) + 0.1 * at
    
    Therefore: v(t) = b * v(t-1) + (1-b) * a(t)
    
 4. v(t) is approximately average over 1/(1-b) days temperature. 
 
 5. This way of computing moving average is good both in terms of memory and computational efficiency. 
 
 6. It's good to compute the moving average for a range of variables. 
