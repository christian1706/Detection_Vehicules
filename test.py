import torch

x = torch.rand(5,2)

print(x)
print(torch.cuda.is_available())