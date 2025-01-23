import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root='C:/Users/garte/image-Afaiwjs6272/train', transform=data_transform)
test_dataset = datasets.ImageFolder(root='C:/Users/garte/image-Afaiwjs6272/test', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 2. Модель CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 класса: настоящие солнце и рисунок

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Инициализация модели и параметров
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Обучение модели и логирование в TensorBoard
writer = SummaryWriter()


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


train_model(model, train_loader, criterion, optimizer, num_epochs=10)


# 4. Генерация эмбеддингов
def get_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            embeddings.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return embeddings, labels


embeddings, labels = get_embeddings(model, test_loader)


def plot_embeddings(embeddings, labels):
    # Применяем PCA для начальной редукции размерности до 2, так как у нас слишком мало признаков
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Применяем TSNE для дальнейшей редукции размерности
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)

    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='jet', s=10)
    plt.colorbar()
    plt.title("TSNE Visualization of Embeddings")
    plt.show()

plot_embeddings(embeddings, labels)


# 6. Transfer Learning: использование предобученной модели ResNet
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TransferLearningModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Инициализация модели transfer learning
transfer_model = TransferLearningModel()
optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)

# Обучение модели Transfer Learning
train_model(transfer_model, train_loader, criterion, optimizer, num_epochs=10)


# 7. Оценка модели
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Выводим отчет о классификации
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    # Строим матрицу ошибок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


evaluate_model(model, test_loader)
evaluate_model(transfer_model, test_loader)
