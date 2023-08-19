import torch
x = torch.arange(60).view(3,2,5,2)
print(x)
x_3=torch.flip(x,dims=[3])
print(x_3)
x_0=torch.flip(x,dims=[0])
print(x_0)