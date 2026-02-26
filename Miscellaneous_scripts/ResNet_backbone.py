import torch
from torchvision import models

model = models.resnet50(weights='DEFAULT')

torch.save(model.state_dict(), "Models/Pretrained/resnet50_weights.pth")
print("Wagi zapisane pomyślnie. Możesz teraz odciąć internet.")