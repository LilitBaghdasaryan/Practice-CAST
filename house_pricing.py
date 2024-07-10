from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd

writer = SummaryWriter(log_dir='runs')

df = pd.read_csv('train.csv')

class HousePriceNet(nn.Module):
    def __init__(self):
        super(HousePriceNet, self).__init__()
        self.fc1 = nn.Linear(288, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc5 = nn.Linear(32, 1)
        # self.fc6 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc4(x)
        # x = self.relu(x)
        # x = self.fc5(x)
        # x = self.relu(x)
        # x = self.fc6(x)
        return x    
  
def categorize_features(data):
    categorical_features = []
    numerical_features = []

    for i in range(len(data.columns)):
        if isinstance(data[data.columns[i]].iloc[1], str):
            categorical_features.append(df.columns[i])
        else:
            numerical_features.append(df.columns[i])

    return categorical_features, numerical_features

def data_imputing(data):
    imp_mean = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    imp_mean.fit(data)
    imputed = imp_mean.transform(data)
    data_imputed = pd.DataFrame(imputed, columns=data.columns)
    return data_imputed   

def feature_transformer(data_imputed, ct):
    fitted = ct.fit_transform(data_imputed)
    features = ct.get_feature_names_out()
    transformed_data = pd.DataFrame(fitted.toarray(), columns=features)
    return transformed_data, features

def preprocessing_data(data, test_set=False):
    data_imputed = data_imputing(data)

    categorical_features, numerical_features = categorize_features(data_imputed)
    if not test_set:
        numerical_features.pop()

    ct = ColumnTransformer(
        [('norm', Normalizer(norm='l1'), numerical_features),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough')
    
    transformed_data, features = feature_transformer(data_imputed, ct)
    features = np.setdiff1d(features, 'remainder__SalePrice')
    return transformed_data, features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocessed_data, features = preprocessing_data(df)  # Keeping the columns' names to add the missing ones to test_set later

target = preprocessed_data['remainder__SalePrice']
features = preprocessed_data.drop(columns=['remainder__SalePrice'])
features_copy = features

# Converting to tensors
features = torch.from_numpy(features.values).float().to(device)
target = torch.from_numpy(target.values).float().to(device)

dataset = TensorDataset(features, target)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=50)

model = HousePriceNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def gradient_decent(dataloader, name, epochs):
    losses = []
    running_loss = 0.0

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.unsqueeze(1).to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()**0.5

            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / 10
                losses.append(avg_loss)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.10f}')
                writer.add_scalar(f'Loss', avg_loss, epoch * len(dataloader) + batch_idx)
                running_loss = 0.0

    return losses

train_losses = gradient_decent(train_loader, 'Train', 2000)

test_dataset = pd.read_csv('test.csv')

preprocessed_test, _ = preprocessing_data(test_dataset, test_set=True)
features_to_add = [col for col in features_copy if col not in preprocessed_test.columns]
preprocessed_test[features_to_add] = 0
preprocessed_test = torch.from_numpy(preprocessed_test.values).float().to(device)
test_ids = test_dataset['Id'].values

model.eval()
with torch.no_grad():
    outputs = model(preprocessed_test)
    outputs = outputs.to('cpu').numpy().flatten()

pred_df = pd.DataFrame({'Id': test_ids, 'SalePrice': outputs})
pred_df.to_csv('HousePricePredicitions.csv', index=False)

def evaluate_validation_set(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # Turn off gradients for validation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()**0.5

    avg_val_loss = val_loss / len(val_loader)
    return f'Validation Loss:       {avg_val_loss:.10f}'

print(evaluate_validation_set(model, val_loader, criterion))



