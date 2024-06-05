## Project Title
R-CONV: An Analytical Approach for Efficient Data
Reconstruction via Convolutional Gradients
## Description
This project contains our paper's code implementations and experiments on advanced gradient-based data leakage methods on convolutional layers. <br>

We present an efficient approach to reconstructing accuratly training examples from gradients and a surprising finding: even with non-fully invertible activation functions such as ReLU, we can analytically reconstruct training samples from the gradients. To the best of our knowledge, this is the first analytical approach that successfully reconstructs convolutional layer inputs directly from the gradients, bypassing the need to reconstruct layers' outputs.<br>

## Methodology

Given that typical networks use convolutional layers followed by fully connected (FC) layers, we start by constructing the first fully connected layer.
1. **Reconstructing FC Layer Input**:
    - We reconstruct the input to the first FC layer as follows: for each point \( x_n \), we divide one of the corresponding gradient weights by the gradient bias for any connected node to this input: $   x_n = \frac{\partial \ell}{\partial w_{nm}} \bigg/ \frac{\partial \ell}{\partial b_m}
    $
    - Additionally, we compute the gradient with respect to the input of the FC layer. For each point \( x_n \), we multiply the corresponding gradient bias by the corresponding weight for each connected node:$
    \frac{\partial \ell}{\partial x_n} = \frac{\partial \ell}{\partial b_m} \times w_{nm}
    $
    - We then sum all these gradients to get the final gradient with respect to this point:
    $
    \frac{\partial \ell}{\partial x_n} = \sum_{c=1}^{C} \frac{\partial \ell}{\partial x_n}
    $
 <br>   
**Key Success Factor**:<br>
    - The success of our approach hinges on using the reconstructed input to propagate the computed gradient through the activation function.
2. **Propagating Gradients**:
    - Given that common activation functions, like sigmoid, tanh, and ReLU, have well-defined mathematical expressions for their derivatives. These derivatives can be calculated efficiently using the activation function's output value (the input that we have constructed),we can construct the gradient w.r.t the ouptut of the previous layer (convloutional layer) through the chain rule:$    \frac{\partial\ell}{\partial O} =  \frac{\partial\ell}{\partial X}\times A’(O) $<br>
    (you can find a table of derivative of most common activation function provided in the paper)
   
3. **Reconstructing convloutinal Layer Input**:
    - Using the gradients w.r.t the ouptput of the convloutional layer that we constructed, we can reconstruct the input of the previous layer using the weight gradients. As each weight gradient is a function of some input points and some output gradients points. for instant:<br> $    \frac{\partial \ell}{\partial w_{d,i}}= \frac{\partial \ell}{\partial o_{d,1}}\times x[r[1]] +\frac{\partial \ell}{\partial o_{d,2}}\times x[r[2]]+\dots+\frac{\partial \ell}{\partial o_{d,m}}\times x[r[m]]  $
    - If the layer has enough filters, we can have a set of linear equations enabling the reconstruction of the input.<br>
    - This methodology highlights the innovative steps we take to leverage gradient information and activation functions to reconstruct inputs in convolutional layers, significantly enhancing the effectiveness of gradient-based data leakage attacks.

## Google Colab
We provide Open [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNDLk53NWFdQHV20S_oHfcLPVyo4jNYB?usp=sharing)
for quick reproduction.
## Usage
You can run the script through:
```bash
python main.py 
```
**Options**  

-d : Choose between three datasets. Options are ["CIFAR10","MNIST","CIFAR100"].<br>
-i : Choose the image index.<br>
-a : Choose the type of activation function. Options are:["ReLU","LeakyReLU","Sigmoid","Tanh"].  

**Example**<br>
Here’s an example command to run the script with the CIFAR100 dataset, using image index 92 and the ReLU activation function
```bash
python main.py -d CIFAR100 -i 92 -a ReLU
```
## Network Architectures
Under the folder **Notebooks_Various_setting** you can find different notebooks for various network architectures available in the repository.<br>
1. **LetNet**: The architecture that has been used in DLG.<br>
2. **LetNet_O**: An optimized version of LetNet, demonstrating that even using fewer filters we can reconstruct the images using gradient constraints.<br>
3. **O_CNN6**: The architecture that has been used in R-GAP, demonstrating that with only 3 filters at the last layer, we can reconstruct the image instead of using 128 filters as demonstrated in R-GAP.<br>
4. **LetNet-4-Relu**: The architecture of LetNet using the ReLU function between the four convolution layers.<br>
5. **LetNet-All-Relu**: The architecture of LetNet using the ReLU function between all the layers except the output layer. By using ReLU in the first layer, it requires a much larger number of filters as the gradient constraints are less in the first layer where the input channels are only 3.<br>

## Intense Testing

Under the folder **Intense_testing** you can find three different notebooks. Each notebook runs the script over the first 100 images of a specific dataset.

### Notebooks

1. **Test_on_MNIST**: This notebook runs the script over the first 100 images of the MNIST dataset.
2. **Test_on_CIFAR10**: This notebook runs the script over the first 100 images of the CIFAR10 dataset.
3. **Test_on_CIFAR100**: This notebook runs the script over the first 100 images of the CIFAR100 dataset.

Each notebook is designed to automate the process of running the script on the respective dataset, providing results and insights based on the first 100 images.