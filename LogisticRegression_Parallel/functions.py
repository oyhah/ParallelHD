import torch
import math

def Ackley(x):
    f1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    f2 = torch.exp(0.5 * (torch.cos(2 * math.pi * x[0]) + torch.cos(2 * math.pi * x[1])))
    f = f1 - f2 + math.e + 20

    return f