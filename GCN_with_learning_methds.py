from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self, dataset=None):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train(optimizer=None, model=None, data=None):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # 只计算node for training部分的loss用于反向传播
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    optimizer.step()


def val(model=None, data=None, epoch=None):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('[Epoch{:5}]    Accuracy:{:.4f}'.format(epoch, acc))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(root='../datasets/Cora', name='Cora')
    data = dataset[0].to(device)
    print(data)

    model = Net(dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        train(optimizer, model, data)
        if epoch % 20 == 0:
            val(model, data, epoch)
