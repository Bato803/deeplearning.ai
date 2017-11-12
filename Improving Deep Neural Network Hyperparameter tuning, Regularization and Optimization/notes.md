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
    
 5. Other techniques
    - Data Augmentation. (Inexpensive way)
    - Early Stopping (According to the error rate of dev set)
    
## Optimization

1. Normalizing input features -> affect your cost function shape(more round and easier to optimize) -> speed up gradient descent.

2. If the elements in weight matrix is a bit larger than one, in a very deep network, the activation might explode. Conversely, if they are less than one, the activation might vanish. Same with the gradients in BP. 

3. Partial solution for vanishing/exploding gradient problem: Careful choice of initilization. Concretely, set the input feature for each layer to be mean zero and standard variance. 

And then: np.randn.random(shape) * np.sqrt(2/n^(l-1)) for Relu activation. 

If you're using tanh, the last term becomes np.sqrt(1/n^(l-1)) (Xavier initilization)
