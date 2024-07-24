from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import pandas as pd
import csv
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1_4 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_8 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv8_16 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv16_32 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1_4(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv4_8(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv8_16(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv16_32(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

model = ConvNet().to(get_device())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

df = pd.read_csv('/home/lilitbaghdasaryan/Desktop/hm/Practice-CAST/digits/train.csv')
dataset = df_to_tensor(df)
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

batch_size = 60
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

epochs = 75
running_correct = 0
running_loss = 0

for epoch in range(epochs):
    for batch_idx, x in enumerate(train_loader):
        labels = x[:, 0]
        labels = labels.long()
        imgs = x[:, 1:]
        imgs = imgs.unsqueeze(-1)
        imgs = imgs.unsqueeze(1)
        imgs = imgs.view(60, 1, 28, 28)
        outputs = model(imgs)
        loss = criterion(outputs.float(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        softmax = nn.Softmax(dim=1)
        y = torch.argmax(softmax(outputs),dim=1)
        running_correct += (y == labels).sum().item()

        if (batch_idx + 1) % 80 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('training_loss', running_loss / 80, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('accuracy', running_correct / (80 * 60) , epoch * len(train_loader) + batch_idx)

            running_loss = 0.0
            running_correct = 0
        
        writer.close()

model.eval()
with torch.no_grad():
    correct = 0
    accuracy = 0
    for x in val_loader:
        labels = x[:, 0]
        labels = labels.long()
        imgs = x[:, 1:]
        imgs = imgs.unsqueeze(-1)
        imgs = imgs.unsqueeze(1)
        imgs = imgs.view(60, 1, 28, 28)
        outputs = model(imgs)

        softmax = nn.Softmax(dim=1)
        y = torch.argmax(softmax(outputs),dim=1)
        
        correct = (y == labels).sum().item()
        accuracy += correct / 60
print(accuracy / len(val_loader))


df = pd.read_csv('/home/lilitbaghdasaryan/Desktop/hm/Practice-CAST/digits/test.csv')

test_set = df_to_tensor(df)
outputs = [['ImageId', 'Label']]

for i, img in enumerate(test_set):
    img = img.unsqueeze(0)
    img = img.unsqueeze(1)
    img = img.view(1, 1, 28, 28)
    output = model(img)
    softmax = nn.Softmax(dim=1)
    y = torch.argmax(softmax(output),dim=1)
    outputs.append([i + 1, y.item()])

with open('predicted_digits.csv', mode='w', newline='') as file:

    writer = csv.writer(file)
    writer.writerows(outputs)