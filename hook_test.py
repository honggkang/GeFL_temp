import torch 
import torch.nn as nn

class myNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3,10,2, stride = 2)
    self.relu = nn.ReLU()
    self.flatten = lambda x: x.view(-1)
    self.fc1 = nn.Linear(160,5)
    self.seq = nn.Sequential(nn.Linear(5,3), nn.Linear(3,2))
    
  def forward(self, x):
    x = self.relu(self.conv(x))
    x = self.fc1(self.flatten(x))
    x = self.seq(x)
  

net = myNet()
visualisation = {}

def hook_fn(m, i, o):
  visualisation[m] = o 

def get_all_layers(net):
  for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
    if isinstance(layer, nn.Sequential):
      get_all_layers(layer)
    else:
      # it's a non sequential. Register a hook
      layer.register_forward_hook(hook_fn)

get_all_layers(net)

  
out = net(torch.randn(1,3,8,8))

# Just to check whether we got all layers
visualisation.keys()      #output includes sequential layers​