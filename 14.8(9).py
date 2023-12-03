import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = Planetoid(root='./data', name='Cora', transform=None)

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(in_channels=dataset.num_features, out_channels=dataset.num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

def train():
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def evaluate(data, model):
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        y_true = data.y[data.test_mask]
        y_pred = pred[data.test_mask]
        accuracy = accuracy_score(y_true.cpu(), y_pred.cpu())
        return accuracy

for epoch in range(100):
    loss = train()
    accuracy = evaluate(data, model)
    print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

print("Training finished!")

model.eval()
with torch.no_grad():
    output = model(data)
    pred = output.argmax(dim=1)
    correct = pred.eq(data.y)
    accuracy = correct[data.test_mask].sum().item() / data.test_mask.sum().item()
    print(f'Test accuracy: {accuracy:.4f}')

pca = PCA(n_components=2)
embedding = pca.fit_transform(data.x.cpu().detach().numpy())

plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=data.y.cpu().numpy(), cmap='jet', s=20)
plt.title('PCA Visualization of Cora Dataset')
plt.colorbar()
plt.show()
