print('Importing libraries.....')
import argparse
import numpy as np
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import torchvision.transforms as T
from skimage.metrics import mean_squared_error, structural_similarity
from skimage.io import imread
from skimage.color import rgb2gray
import time
from utils import *
from functions import *
torch.manual_seed(50)
torch.set_default_dtype(torch.float64)
parser = argparse.ArgumentParser(description="R-CONV- Attack on convloution layers")
parser.add_argument("-d", "--dataset", help="Choose the dataset.", choices=["CIFAR10", "MNIST","CIFAR100"], default="CIFAR100")
parser.add_argument("-i", "--index", help="Choose the index of the image to reconstruct.", type=int, default=25)
parser.add_argument("-a", "--activation_function", help="choose the activation function between the layers.", choices=["ReLU","LeakyReLU","Sigmoid","Tanh"], default="ReLU")
args = parser.parse_args()

dataset_class=getattr(datasets,args.dataset)
dataset_type=args.dataset
image_index=args.index
act_function=args.activation_function

if (dataset_type=='CIFAR100'):
    number_classes=100
    number_channels=3
elif(dataset_type=='CIFAR10'):
    number_classes=10
    number_channels=3
elif(dataset_type=='MNIST'):
    number_classes=10
    number_channels=1
if act_function=='ReLU':
    act=nn.ReLU()
elif act_function=='LeakyReLU':
    act=nn.LeakyReLU(negative_slope=0.2)
elif act_function=='Tanh':
    act=nn.Tanh()
elif act_function=='Sigmoid':
    act=nn.Sigmoid()
act_0= nn.LeakyReLU(negative_slope=0.2)

print("Downloading Dataset.......")
dst = dataset_class("~/.torch", download=True)
tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])
tt = transforms.ToPILImage()

device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
# print("Running on %s" % device)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.body = nn.Sequential(
        nn.Conv2d(number_channels, 12, kernel_size=5, padding=5//2, stride=2,padding_mode='zeros'),
        act_0,  
        nn.Conv2d(12, 16, kernel_size=5,padding=5//2, stride=2,padding_mode='zeros'),
        act_0,
        nn.Conv2d(16, 12, kernel_size=5, padding=5//2, stride=1,padding_mode='zeros'),
        act,
        nn.Conv2d(12, 3, kernel_size=5, padding=5//2, stride=1,padding_mode='zeros'),
        act,
        )
        self.fc = nn.Sequential(
            nn.Linear(192, number_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out,x
net = LeNet().to(device)
net.apply(weights_init)
criterion = cross_entropy_for_onehot



######### Feed the image to the network and compute gradients #########
img_index = image_index
gt_data = tp(dst[img_index][0]).to(device)
gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes=number_classes)

out,org_x = net(gt_data)
y = criterion(out, gt_onehot_label)

dy_dx = torch.autograd.grad(y, net.parameters())

# Extract the gradients and initial parameters
original_dy_dx = [_.numpy() for _ in dy_dx]
param = [i.detach().numpy() for i in net.parameters() if i.requires_grad]
#---------------------------------------------------------------#


#---------------------------------------------------------------------#
print("*"*20)
start_time=time.time()
print('R-CONV start .....')
print("*"*20)

# Reconstrut the inpt and gradient w.r.t input of the First Fully connected layer
cnn_layers,fc_layers=inspect_model(net)
FC=fc_layers[0]
in_dim=FC['input_dim']
out_dim=FC['output_dim']
w_index=FC['param_index']
b_index=w_index+1
#Compute The gradient w.r.t input to FC sum ( bias * weight)
FC_input_gradient=np.zeros(in_dim)
FC_input_values=np.zeros(in_dim)
for i in range(out_dim):
    for n in range(in_dim):
      FC_input_gradient[n]+= original_dy_dx[b_index][i]*param[w_index][i][n] 
# Compute the values of the input of FC ( weigh/bias)
for n in range(in_dim):
  for k in range(out_dim):
    if original_dy_dx[b_index][k]!=0:
      FC_input_values[n]= original_dy_dx[w_index][k][n]/original_dy_dx[b_index][k]
Computed_gradients= torch.tensor(FC_input_gradient.copy())
Computed_values=FC_input_values.copy()
print(f"- The input and the gradient w.r.t the input of the FC layer successfully reconstructed")
#--------------------------------------------------------------------------------#

# Reconstruct the input and graident w.r.t to input of the convlaytional layers:<br>
# 1- Propagte the pre computed gradient of the subsequent layer through the activation function <br>
# 2- Construct input based on the gradient constrains <br>
# 3- Drive the graident w.r.t the input

for n in range(len(cnn_layers)-1,-1,-1):
  # Extract the layer setting 
  cnn=cnn_layers[n]
  num_filters=cnn['number_filters']
  num_c=cnn['input_channel']
  dim_x=cnn['input_dim']
  w_index=cnn['param_index']
  weight_gradient=original_dy_dx[w_index]
  output_gradient= Computed_gradients
  padding=cnn['padding']
  stride=cnn['stride']
  out_dim=cnn['output_dim']
  act_fun=cnn['act_fun']
  kernal=cnn['kernal']
  
  
  #--------Propagate the gradient through the activation Funciton------------#
  Computed_values=Computed_values.reshape(out_dim**2*num_filters)
  output_gradient=output_gradient.reshape(out_dim**2*num_filters)
  output_values=Computed_values.reshape(out_dim**2*num_filters)
  for i in range(out_dim**2*num_filters):
    if(act_fun=='ReLU'):
      if np.round(Computed_values[i],7)<=0:
        output_gradient[i]=0
    elif(act_fun=='LeakyReLU'):
      if np.round(Computed_values[i],7)<0:
        output_gradient[i]=output_gradient[i]*0.2
        output_values[i]=output_values[i]/0.2
    elif(act_fun=='Sigmoid'):
        output_gradient[i]=Computed_values[i]*(1-Computed_values[i])*output_gradient[i]
    elif(act_fun=='Tanh'):
        output_gradient[i]=(1-Computed_values[i]**2)*output_gradient[i]
#--------------------------------------------------------------------------------#
  
  
  if(n!=0):   # check if reached the first convloution layer        
    output_gradient=output_gradient.clone().detach()
    output_gradient=output_gradient.reshape(num_filters,out_dim,out_dim)

    #-----construct the input_values of the layer using gradinet constrains-------#
    x=construt_input_using_gradients(num_filters,num_c,dim_x,output_gradient,weight_gradient,padding,stride,kernal)
    #------------------------------------------------------------------------------#
    
    #--------Compute the gradient w.r.t input of the layer------------------------#
    dL_dX_CNN= drive_gradient(x.shape,param[w_index],output_gradient,stride,padding)
    #------------------------------------------------------------------------------#
    
    Computed_gradients= dL_dX_CNN
    Computed_values=x
    print(f"- The input and the gradient w.r.t the input of the {n} CNN layer successfully reconstructed")
  else:      # in case of the first convloution layer we construct the input using weight constrains
    Y=output_values.reshape(num_filters,out_dim,out_dim)
    weights = param[w_index] 
    bias=param[w_index+1]
    for i in range(num_filters):
        Y[i]=Y[i]-bias[i]
        
    #-----construct the input_values of the first layer using weights constrains-------#  
    sol= construt_input_using_weights(num_filters,num_c,dim_x,Y,weights,padding,stride,kernal)
    #------------------------------------------------------------------------------#

# View Reconstructed Image
end_time=time.time()
reconstructed_val=torch.tensor(sol).reshape(number_channels,32,32)
oringinal_val=org_x.clone().detach()
oringinal_val=oringinal_val.reshape(number_channels,32,32)
plt.figure(figsize=(10,5))
plt.subplot(1, 2,  1)
plt.imshow(tt(oringinal_val))
plt.title("Original")
plt.axis('off')
plt.subplot(1, 2,  2)
plt.imshow(tt(reconstructed_val))
plt.title("reconstructed")
plt.axis('off')
plt.savefig('reconstruction.png')


#measure the quality of the image

mse=mean_squared_error(np.array(oringinal_val),np.array(reconstructed_val))
print("*"*30)
print("Both images have been saved to 'reconstruction.png'.")
print(f'The Error in the construction(MSE): {mse}')
Max=255
PSNR = 20*np.log10(Max)-10*np.log10(mse)
print("PSNR: ",PSNR)
print("*"*30)
print(f"Reocnstruction_time= {end_time-start_time:.2f} s")

plt.show()
