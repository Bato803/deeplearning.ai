## Convolutional Neural Network


## Detection Algorithm

### Object Localization

1. In this content, we just localize one object. 
2. Simply increase the number of output neurons for CNN's last layer, the output neurons will output 
    - whether it's an object. (y1)
    - x, y, w, h
    - p(class1), p(class2), ...
3. Cost function:
    - If there is an object, sum of square of all different elements. 
    - If there is no object, loss is just (y1-y_hat)^2

### Landmark Detection

1. Let the NN final layer have more neurons to output the coordinates of the landmarks. 
2. The landmarks of the training images should be consistant. 

### Base Line CNN object detection - Sliding windows detection 

1. Say in car detection, we can first train a CNN in classifying if the input is a car. 
2. Slide a window across the whole image, feed the window's content into CNN. 
3. Use windows of different sizes to get scale invariance. 

**Disadvantages: large stride-> too much distracting content; small stride-> high computational cost.**

### Convolutional Implementation of Sliding Windows

1. Convert fc layer into conv layer. 
    - (5,5,16) --FC-- 400 neurons. 
    - (5,5,16) -- 400 filters of size(5,5,16) -- vector of length 400. 

2. Overfeat idea. 
    - Putting different areas into the conv net has many duplicate computation, we need to share those computation. 
    - Convert FC layer into conv layer. Extract features from different regions of feature maps.  

### Yolo algorithm in bounding box prediction . [Link](https://www.coursera.org/learn/convolutional-neural-networks/lecture/9EcTO/bounding-box-predictions)

1. Define a grid on the image. (say a 3 by 3 grid, and the image are split into 9 regions.)
2. For each grid cell, run the object localization algorithm. (A conv net outout the coordinates of bounding box, probability of an object and the probability of each class.)
3. After step 2, we have an output vector for each grid cell. Then we assigned the detected object to the grid cell where the mid point of this object reside. Each cell will output one vector. 

**Advantages: By examining the outputs volumn, we know the coordinates of the bounding boxes, whereas the overfeat idea can't provide us with the exact location of the object.**

**Disadvantage: Multiple objects in a grid cell. (>.<)   An object is split into two cells. (>.<)**

### Intersection over union. 

1. IoU is a measure of the overlap between two bounding boxes. If detection is perfect, IoU = 1
2. If IoU >= 0.5, treated as correction localization. This way, we could use accuracy to measure our detection task. 

### Non-maximum suppression. 
