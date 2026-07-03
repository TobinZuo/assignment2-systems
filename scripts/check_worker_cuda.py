import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
