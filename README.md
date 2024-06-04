## Google Colab
We provide Open [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNDLk53NWFdQHV20S_oHfcLPVyo4jNYB?usp=sharing)
for quick reproduction.
## Usage
Reconsturction of the 92-th image in CIFAR100 where ReLU activation function is employed
```bash
python main.py -d CIFAR100 -i 92 -a ReLU

Options
-d : Choose between three datasets. Options are ["CIFAR10", "MNIST","CIFAR100"].
-i : Choose the image index.
-a : Choose the type of activation function. Options are ["ReLU","LeakyReLU","Sigmoid","Tanh"].
Example
Above is an example command to run the script with the CIFAR100 dataset, using image index 92 and the ReLU activation function:
```
## Intense Testing

This section provides details on how to perform intense testing using three different notebooks. Each notebook runs the script over the first 100 images of a specific dataset.

### Notebooks

1. **Test_on_MNIST**: This notebook runs the script over the first 100 images of the MNIST dataset.
   - [Open Test_on_MNIST Notebook]

2. **Test_on_CIFAR10**: This notebook runs the script over the first 100 images of the CIFAR10 dataset.
   - [Open Test_on_CIFAR10 Notebook]

3. **Test_on_CIFAR100**: This notebook runs the script over the first 100 images of the CIFAR100 dataset.
   - [Open Test_on_CIFAR100 Notebook]
 

Each notebook is designed to automate the process of running the script on the respective dataset, providing results and insights based on the first 100 images.