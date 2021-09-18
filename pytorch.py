import torch
cpu_tensor = torch.zeros(2, 3)
device = torch.device("cuda:0")
gpu_tensor = cpu_tensor.to(device)
print(gpu_tensor)
print(torch.version.cuda)