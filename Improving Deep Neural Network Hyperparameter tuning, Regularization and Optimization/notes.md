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
 
 
 ### Mini-batch gradient descent
 1. Using SGD might cause you losing the speedup from vectorization. 
 
 2. Choosing Mini-batch size rule:
    1. If small training set(<2000), just use batch GD.
    2. Typical mini batch size: 64, 128, 256, 512.
    3. Make sure mini-batch size fits into your CPU or GPU.
 
 
 ### Some math
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
 
 7. Bais Correction for Exponentially weighted Average:
    - v(t) = (b * v(t-1) + (1-b) * a(t))/(1-b^t)
    
 ### Gradient Descent with Momentum
 1. Basic idea: Use exponentially weighted average to compute the gradient, and use that gradient to update the parameters. 
 
 2. Momentum:
    - On iteration t, compute dw, db on current mini-batch. 
        - Vdw = beta * Vdw + (1-beta) * dw
        - Vdb = beta * Vdb + (1-beta) * db
        - w = w - alpha * Vdw
        - b = b - alpha * Vdb
    - It smooths out oscillation in the direction that we don't need. Ant at the same time maintaining the gradient that points toward the minimum. 
    - The most commonly use beta is 0.9, which means we're averaging out the last ten gradients. 
    - When implementing gradient descent with momentum, it's not very often to use bias correction. 
    - Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
    
### RMSprop
1. On iteration t, compute dw, db on the current mini-batch
    - Sdw = beta2 * Sdw + (1-beta2) * (dw)^2   <-- element-wise square
    - Sdb = beta2 * Sdb + (1-beta2) * (db)^2
    - Keeping a exponentially average of the square of the derivative. 
    - w = w - alpha * dw /(sqrt(Sdw))
    - b = b - alpha * db/(sqrt(Sdb))
    - Intuition:
        - If during training, the derivative in the unwanted direction is large(say db is large)
        - Then in the updating equation, we are dividing a relatively large number. 
        - And that helps damp out the oscillation in the unwanted direction. 
        - The derivative in the wanted direction would keep going. 
    
### Adam
1. Basically it's putting momentum and RMSprop together. 
2. Initilization:
    - Vdw=0, Sdw=0, Vdb=0, Sdb=0
    - On iteration t, compute dw, db using current mini-batch
        - Momentum
        - Vdw = bete1 * Vdw + (1-beta1) * dw
        - Vdb = beta1 * Vdb + (1-beta1) * db
        - Momentum with correction
        - Vdw(corrected) = Vdw/(1-beta1^t)
        - Vdb(corrected) = Vdb/(1-beta1^t)
        - RMSprop
        - Sdw = beta2 * Sdw + (1-beta2) * (dw)^2
        - Sdb = beta2 * Sdb + (1-beta2) * (db)^2
        - Sdw(corrected) = Sdw/(1-beta2^t)
        - Sdb(corrected) = Sdb/(1-beta2^t)
        - Update
        - w = w - alpha * (Vdw(corrected))/(sqrt(Sdw(corrected)+epsilon))
        - b = b - alpha * (Vdb(corrected))/(sqrt(Sdb(corrected)+epsilon))
        
3. Hyper-parameter
    - beta1: 0.9(default for moving average)
    - beta2: 0.999 (dw^2)
    - epsilon: 10^(-8)
    - alpha: needs to be tuned. 
    
4. Adam - Adaptive moment estimation
    Some advantages of Adam include:
    - Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
    - Usually works well even with little tuning of hyperparameters (except  αα )


### Learning rate decay
1. As alpha goes smaller, the steps are smaller. Ends up in a tiny region around the minimum. 
2. 1 epoch - 1 pass through the data. 
3. alpha = 1/(1+decay_rate * epoch_num) * alpha0
4. alpha = 0.95^(epoch_num) * alpha0
5. alpha = k/(epoch_num) * alpha0

### Local Optima
1. Most of the local optima in deep learning cost function is saddle point. 
2. It takes a very very long time to go down to the saddle point before it finds its way down. (Problem of plateaus)
3. Use momentum or Adam to solve this kind of problems. 
        
