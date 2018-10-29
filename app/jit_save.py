import torch
from data import MIO_CLASSES
from ssd import build_ssd
import torch.onnx
import numpy as np
import pickle
dummy_input = torch.randn(1, 3, 300, 300)

net = build_ssd('onnx', 300, len(MIO_CLASSES) + 1)
net.load_state_dict(torch.load('weights/MIOSTAT.pth'))
net.eval()
pickle.dump(net.priors.numpy(),open('priors.pkl','wb'))
traced_script_module = torch.jit.trace(net, dummy_input)
traced_script_module.save("model.pt")


mod = torch.jit.load('model.pt')
print(len(mod(dummy_input)))