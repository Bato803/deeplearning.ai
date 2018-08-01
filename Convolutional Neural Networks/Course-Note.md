# Convolutional Neural Network

## Case Study

### Practical Advice

1. Data Augmentation. 
  - Random cropping, mirroring, shearing, rotation, ...
  - Color shifting (take different R,G,B values to distort image channels) 
  - PCA color augmentation. (keep the overall color of the tints the same.)

2. State of Computer Vision
  - Few tips for doing well on benchmarks/competitions 
    1) Emsembling - averaging different NNs outputs
    2) Multi-crop at test time (Run classifier on multiple versions of test images, and average results)

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

1. Say, in a 19 by 19 grid, we have a (19, 19, 8) volumn. (pc, bx, by, bw, bh, c1, c2, c3)
2. Discard the bounding boxes with pc less than some threshold, such as 0.6
3. Pick the box with the highest pc * c1, treated as ouput, and discard remaining bounding boxes with >= 0.5 IoU. 
4. Repeat step 3 for the other two classes. 

### Anchor Boxes

A grid cell that can detect multiple object. 

1. Pre-defined boxes with different shape. 
2. Say if we have two different boxes, then we associate our output volumn with different anchor boxes. In this case, our output will have the shape (19,19,16) instead of (19,19,8)

Without anchor boxes, each object is assigned to grid cell that contains that object's midpoint. 
With anchor boxes, each object is assigned to grid cell that contains that object's midpoint and anchor box for the grid cell that has higher IoU with ground truth. 

## Face Recognition

### One shot learning
1. Learning from one example to recognize the person again.
2. Solution: Learning a similarity function. 
  - d(img1, img2) = degree of difference between images. 
    - If d(img1, img2) <= threshold, same person, if d(img1, img2) > threshold, different person. 
    - This could be used to solve face verification problem. 
  - For recognition task:
    - Compare input image with the images in the database. 
    - Figure out the person with minimum d. 

### Siamese Network
1. Feed an input image into CNN, get the encoding vector f(x1) of input image from CNN.
2. Feed second input image into the same CNN with the same parameters, and get the encoding vector f(x2) from it. 
3. If the encoding is good, find the d(x1, x2) = || f(x1) - f(x2) || ^ 2
4. Parameters of NN define an encoding f(xi)
  - Goal of training this network, it's if xi and xj are the same person, then ||f(xi)-f(xj)|| is small. 

### Triplet Loss function 
Triplet loss function is used to train Siamese Network to learning a good encoding of input image. 
1. Anchor image, positive image(same), negative image(different)
2. Want: ||f(A)-f(P)|| < || f(A)-f(N)||
3. To prevent f always predict zero, we added margin alpha to the above function:
  - ||f(A)-f(P)|| - || f(A) - f(N) || + alpha <= 0 (alpha>0)


## Neural Style Transfer

### What are deep conv net lelarning? 
Method: Pick a unit in a particular layer, and find the image patches that maximize the unit's activation. Repeat for other units. 
  - For example, units in the first layer are looking for simple features such as edges of particular color. 
  - Units in layer two might be looking at more complex features, such as vertical textures of lots of vertical line, rounded shape in the left side of the image, thin vertical lines. 
  - In layer three, features are getting more complicated, such as tires at the lower left corner, detecting people, square or honey shape texture. 

### Cost Function
- Goal: Given a Content image C and a Style image S, to generate a new image G. 
- Define a cost function J(G) to measure how good is this image. 
- J(G) = alpha * J_{content}(C, G) + beta * J_{style}(S, G) --> how similar is C & G, and how similar is S and G. 
- Steps:
  1. Initiate G randomly. 
  2. Use gradient descent to minimize J(G)

### Content Cost Function
- Say we use hidden layer l to compute content cost. 
- Use pre-trained ConvNet
- Compute activation[l](C) and activation[l](G) of layer l on the images. 
- If activation[l](C) and activation[l](G) are similar, both images have similar content. 
- J_{content}(C,G) = 0.5 * ||a[l](C)-a[l](G)||^2
- Element-wise sum square difference between these two vectors. 

### Style Cost Function

- Define the 'style' of an image: Define style as correlation between activations across channels. 
- Intuitions: Correlation means which/how often the high level texture components tend to occur or not occur together. 
- Use the degree of correlation between channels as a measurement of style. And compare the degree of correlation between generated image's feature map and style image's feature map. 

- Style matrix:
  - G[l] -> (nc, nc)   (nc--number of channels)
  - G[l]{k, k'} = sum_{i} sum{j} a[l](i,j,k) * a[l](i, j, k')
  - If they are correlated, the number would be large and if it's uncorrelated, the number would be small. 
  - J(S, G) = || G[l](S) - G[l](G) || ^ 2
  
### 1D and 3D Generalization of Conv net. 



