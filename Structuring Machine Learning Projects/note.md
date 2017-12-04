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
  

