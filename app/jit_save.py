import torch
from data import MIO_CLASSES
from ssd import build_ssd
import torch.onnx
import numpy as np
import pickle
import time

dummy_input = torch.randn(1, 3, 300, 300)
dummy_input2 = torch.randn(1, 20, 19, 19)

net = build_ssd('onnx', 300, len(MIO_CLASSES) + 1)
net.load_state_dict(torch.load('../weights/MIOODF_OF.pth'))
net.eval()
pickle.dump(net.priors.numpy(),open('priors.pkl','wb'))
traced_script_module = torch.jit.trace(net, (dummy_input, dummy_input2))
traced_script_module.save("model_odf.pt")


mod = torch.jit.load('model_odf.pt')
print(len(mod(*(dummy_input, dummy_input2))))

for _ in range(10):
    mod(dummy_input, dummy_input2)

s = time.time()
for _ in range(100):
   mod(dummy_input, dummy_input2)

print("ODF timing", time.time() - s)
