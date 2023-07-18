import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn

class OpenEyesClassificator():
  class BatchedConvNetworkPad(nn.Module):
    def __init__(self, input_shape=24*24, num_classes=2, input_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3), padding=1), # size = (24 * 24 * 20)
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # (12 * 12 * 20)

            nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = (2, 2)), # size = (11 * 11 * 40)
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # size = (5 * 5 * 40)

            nn.Conv2d(in_channels = 40, out_channels = 60, kernel_size = (2, 2)), # size = (4 * 4 * 60)
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # (2 * 2 * 60)

            nn.Flatten(),
            nn.Linear(240, 128),
            nn.BatchNorm1d(128, 0.7),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, 0.7),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16, 0.7),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes)
        )

    def forward(self, inp):
        out = self.model(inp)
        return out

  def __init__(self, path_mod, device = 'cpu'):
    self.model = self.BatchedConvNetworkPad(input_shape = (24, 24), num_classes = 2)
    self.model.load_state_dict(torch.load(path_mod, map_location=torch.device(device)))
    self.model.to(device)
    self.device = device
    self.transform = transforms.Compose([
        transforms.Resize((24, 24)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(0.6117, 0.1954)
    ])

  def predict(self, inplm):
    with torch.no_grad():
      img = Image.open(inplm)
      img = self.transform(img).to(self.device)
      self.model.eval()
      out = self.model(img.unsqueeze(0))
      out = torch.nn.Softmax(dim=1)(out.cpu())
      return out[0][1].item()

