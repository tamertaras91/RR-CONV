import torch
import torch.nn as nn
import torch.nn.functional as F

def label_to_onehot(target, num_classes):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target
def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-.5, .5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
        

# function to return the details of the layers (e.g. input_dim,num_filters,stride,...)
def inspect_model(model,x_dim=32):
    conv_layers = []
    fc_layers = []
    parmeter_index=0   # Hold the index of the parameters of each layer
    input_dim=x_dim
    input_dim_fc=0
    act_fun=''
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            output_dim=int((input_dim+2*module.padding[0]-module.kernel_size[0])/module.stride[0])+1
            layer_details = {
                'param_index':parmeter_index,
                'input_channel': module.in_channels,
                'number_filters': module.out_channels,
                'stride': module.stride[0],
                'padding': module.padding[0],
                'kernal': module.kernel_size[0],
                'input_dim':input_dim,
                'output_dim':output_dim,
                'act_fun':act_fun
            }
            conv_layers.append(layer_details)
            input_dim=output_dim
            input_dim_fc=input_dim**2*module.out_channels
            parmeter_index+=2
        elif isinstance(module, nn.Linear):
            layer_fc_details = {
                'param_index':parmeter_index,
                'input_dim':(input_dim_fc),
                'output_dim':module.out_features
            }
            fc_layers.append(layer_fc_details)
            input_dim_fc=module.out_features
            parmeter_index+=2
        elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            act_fun=str(module.__class__).split(".")[-1].split("'")[0]
            conv_layers[-1]['act_fun']=act_fun
    return conv_layers,fc_layers
