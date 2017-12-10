## Orthogonalization

- What to tune in order to achieve what effect. 
- Chain of Assumption in ML:
  - Fit well in training set. (Bigger network, optimization algorithms)
  - Fit dev set well. (Regularization, Bigger training set)
  - Fit test set well. (Bigger dev set)
  - Fit real world well. (Cost function, change dev set)
- Tend not to use early stopping. (It makes you fit less well in the training set, and It affects the performance in Training and dev test at the same time. And it's not orthogonal.)


## Set up your Goal
- Single number evaluation metric
  - Precision vs Recall ----> F1 score (Harmonic mean between precision and recall. 2/(1/P + 1/R)
- Satisfying and Optimizing metric.
  - Combining the Accuracy and Running time -> Linear weighted sum? 
  - Or maximize accuracy but subject to 100 ms. 
  - N metric - 1 to optimize, N-1 to be satisfying. (Maximize accuray, and subject to at most 1 false positive every 24 hours operation)
  
  
## Train/Dev/Test distribution
- Random shuffle data before dev/test split. Make sure Dev and Test come from the same distribution.
- Choose dev and test set to reflect data you expect to get in the future. 
- Old way of splitting data:
  - 60% Training, 20% Dev, 20% Test. 
  - Good approach when you have data size 1000/10,000 examples. 
- But now we have 1,000,000 training examples. 
  - 98% Training, 1% Dev(10,000), 1% Test. 
  - In modern DL, it's ok to have a much smaller fraction in Dev, Test. 


## When to change Dev/Test set and metrics
- Add your own preference in your evaluation metrics!

## Comparing with human level performance
- Bayes Optimal Error: Best optimial error. (Hard to surpass)
- If ML is worse than human, you can:
  - Get labelled data from human. 
  - Gain insight from manual error analysis. 
  - Better analysis of bias and variance. 

## Avoidable Bias
- If human level is 1%, but training error is 8% and dev error is 10%, then focus on reducing bias. 
- If human level is 7.5% error, training error 8% and dev error is 10%, focus on reducing variance. 
- At some task, like cat classification, think of human level error as bayes error. (Human is good at CV)
- Avoidable Bias: Difference between bayes error and training error. (In the first case, there is much more potential in reducing avoidable bias, while on the second case, there is more potential in reducing variance)

## Understanding Human Level Performance
- Human level error as a proxy for Bayer error. 
- Choose variance reduction technique or bias reduction technique according to the difference between human level performance and training performance/testing performance. 
- For task that human can do quite well, use human level error as a proxy as bayes error. And try to compare it with your algorithm in Training/Dev set to decide if you should focus on Bias or Variance problem. 
- For tasks that are not natural perception task, it's relatively easier to surpass human level performance. 


## Improving your model
- Two fundamental assumptions of supervised learning. 
  - You can fit training set really well. (Low avoidable bias)
  - The training set performance generalized pretty well to dev/test set 
- Look at the difference between training error and human-level(Bayes) error to decide how much better you should do on your training set. 
- Look at the difference between dev error and trainign error to see how much variance you have. In other words, how much harder you should be working on performance to generalized from training set to dev set. 
- For avoidable Bias: 
  - Train bigger model. 
  - Train longer/ Better optimization algorithms.
  - NN architecture or hyperparameters.
- For variance problem:
  - More data. 
  - Regularization(Data Augumentation, Dropout, L2 regularization...)
  

## Error Analysis


### Carrying out error analysis
If your algorithm is not working as good as you expect, let's do error analysis to see what to do next. 
- Ceiling of a solution. (If you solve this problem perfectly, how much improvement do you got?)
- Find a set of mislabelled example of the dev set, look at those example for either false positive, and false negative. And count up errors for each category. 
- Find out what cause error the most, and prioritize the direction to go. 

### Clearning up incorrectly labelled data. 
Look at three numbers:
1. Overall dev set error. 
2. Errors due to incorrect labels. 
3. Errors due to all other causes. 

But rememeber, if we fixed the labels in dev set:
- Fix the labels for test set as well. 
- Consider examining examples that our algorithm got right. 
- Training set may come from slightly different distribution than test/dev. That's prabobly ok. But test and dev set must come from the same distribution!

### Build your first system quickly and iterate 
- Set up dev/test set and metric
- Build initial system quickly. 
- Use Bias/Variance analysis and error analysis to prioritize next step. 


## Mismatched training and dev/test set
- Training data comes from different data? It's actually ok. 
  - Option One: Randomly shuffle data from different distribution, and then put them into Train/Dev/Test. (Some all split comes from same distribution; but your end goal might be only one of those distributions instead of all of them) This might not be a good option because we are setting up the dev set that we does not want to optimize on. 
  - Option Two: Setting the dev and test set come only from target distribution. (Disadvantage: Training set has different distribution) This approach is better. 
  
  
### Bias and Variance with mismatched data distribution
The way we analyze bias and variance is different when our training example comes from different distribution than test and dev set. 

- If human error of a task is about 0%, our classifier training error is 1%, and dev error is 10%. 
  - If training and dev comes from the same distribution, then our classifier might has a high bias problem. 
  - But what if our dev comes from a different distribution??
  - Ans: we can specifically curve out a 'training-dev' set from the training set, and the training and training-dev set would have same distribution. And perform no training on training-dev set. 
  - If the error on training set and training-dev set has a big difference, then our algorithm has a high variance problem. 
  - If the error on training set and training-dev set is similar, but it jumps high when it comes to dev set. That would be a data mismatch problem. 
  - If the error on training set, training-dev, dev set is similar, but they are all worse than the human performance. That could be a bias problem. 
  
- Principal:
  - Human level error. 
  - Training set error. 
  - Training-dev set error. 
  - Dev set error. 
  - Look at the four quantity above to see if it's bias/variance/data mismatch problem. 
  
- (Human level error <-> Training set error)  Avoidable bias problem. 
- (Training set error <-> Training-dev set error)  Variance problem. 
- (Training dev set error <-> Dev set error) Data mismatch problem. 
- *If there is a huge different between dev and test set error, we over tune to the dev set* 

 ### Address data mismatch problem
- Carry out manual error analysis to try to understand difference between training and dev/test set. 
- Make the training data more similar, and collect more similar data to dev/test set. 
  - Artificial data synthesis. 
  
  
  
  ## Learning from Multiple Tasks
  
  ### Transfer Learning
  
  - If the new data set is small and similar to the original training data:
    - slice off the end of the neural network
    - add a new fully connected layer that matches the number of classes in the new data set
    - randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
    - train the network to update the weights of the new fully connected layer
  
  - If the new data set is small and different from the original training data: (Just use lower level features)
    - slice off most of the pre-trained layers near the beginning of the network
    - add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set
    - randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
    - train the network to update the weights of the new fully connected layer
    
  - If the new data set is large and similar to the original training data:
    - remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
    - randomly initialize the weights in the new fully connected layer
    - initialize the rest of the weights using the pre-trained weights
    - re-train the entire neural network
    
  - If the new data set is large and different from the original training data:
    - remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
    - retrain the network from scratch with randomly initialized weights
    - alternatively, you could just use the same strategy as the "large and similar" data case

### Multi-task Learning

- Instead of using softmax loss function, who predict one label for an example, we can summing up the logistic loss:
  - 1/(m) * sum_i^m * sum_j^4 * (L(y_j^(i), y_j^(i)))

- When multi-task learning makes sense. 
  - Training on a set of tasks that share lower level features. 
  - Amount of data you have for each task is quite similar. 
  - Can train a big enough network to do well on all tasks. 
  
  
## End to End deep learning

### WHAT
- Bybass those hand designed phase, and it really simply the design of the system. 
- Not always work, need a lot of data. 

### WHEN TO USE AND WHEN NOT TO
- Pros of end-to-end learning:
  - Let the data speak instead of having to enforcing human pre-conception. 
  - Less hand-designing of components needed. 
- Cons:
  - Need a large number of data. 
  - Exclude potentially useful hand-designed components.
- Key question:
  - Do you have enough data to learn a function of the complexity needed to map x to y. 
