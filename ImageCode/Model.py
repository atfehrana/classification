# Librairies
import torch.nn as nn

# Creating a personnal model
class CNN(nn.Module):
	#  Architecture model
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.convLayer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding = 1)
        self.convLayer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1)
        self.maxPool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.convLayer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride = 1, padding = 1)
        self.convLayer4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride = 1, padding = 1)
        self.maxPool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.convLayer5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1)
        self.convLayer6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride = 1, padding = 1)
        self.maxPool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(524288, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.maxPool1(out)
        
        out = self.convLayer3(out)
        out = self.convLayer4(out)
        out = self.maxPool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        
        return out