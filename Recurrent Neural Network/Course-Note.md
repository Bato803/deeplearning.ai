# Recurrent Neural Network 

## RNN Model 

### Why not a standard network
1. Inputs, outputs can be different lengths in different examples. 
2. Standard NN doesn't share features learned across different position of text. (Unlike CNN, who share parameters across different position) 

### Language Model and Sequence Generation 
1. First step: Tokenize the sentences. (Form a vocabulary of the words, and then map each of those words to vectors) 
  - Common things: add token of End Of Sentence; ignore the punctuation; tokenize the uncommon words as UNK(unknown word); 
2. Build RNN architecture. 

### Sampling novel sequence
- Character level model pros: don't have to worried about unknown words. 
- Cons: End up with much longer sentence, don't quite good at handling long range dependency. Expensive to train. 

### Vanishing Gradient Problem
