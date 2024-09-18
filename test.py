import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. تحميل البيانات
data = pd.read_csv('C:\\Users\\NITRO\\Desktop\\AI Proj\\Car Features and MSRP\\archive\\data.csv')

data

# 2. معالجة البيانات (اختيار الأعمدة المهمة)
features = ['Make', 'Year', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 
            'Driven_Wheels', 'Number of Doors']
target = 'MSRP'

# تحويل الميزات النصية إلى قيم رقمية باستخدام LabelEncoder
le_make = LabelEncoder()
data['Make'] = le_make.fit_transform(data['Make'])

le_transmission = LabelEncoder()
data['Transmission Type'] = le_transmission.fit_transform(data['Transmission Type'])

le_driven = LabelEncoder()
data['Driven_Wheels'] = le_driven.fit_transform(data['Driven_Wheels'])

# إزالة القيم المفقودة
data = data.dropna(subset=features + [target])

X = data[features].values
y = data[target].values

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. تحويل البيانات إلى PyTorch Tensors
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 4. إنشاء Dataset مخصص
class CarPriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# إعداد DataLoader
train_dataset = CarPriceDataset(X_train, y_train)
test_dataset = CarPriceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. بناء نموذج CNN
class CarPriceModel(nn.Module):
    def __init__(self):
        super(CarPriceModel, self).__init__()
        self.fc1 = nn.Linear(len(features), 128)  # عدد الميزات
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # طبقة التنبؤ

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CarPriceModel()

# 6. تعريف الخسارة والمحقق
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. تدريب النموذج
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 8. اختبار النموذج
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        test_loss += loss.item()

    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
