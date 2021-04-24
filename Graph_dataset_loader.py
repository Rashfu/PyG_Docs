from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean

dataset = TUDataset(root='../datasets/ENZYMES', name='ENZYMES', use_node_attr=True).shuffle()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(dataset.data)

for data in loader:
    print(data)

    x = scatter_mean(data.x, data.batch, dim=0)
    print(x.size())
# train_dataset = dataset[:540]
#
# test_dataset = dataset[540:]
#
# print("train dataset 90% || test dataset 10%")
#
# print(train_dataset)
# print(test_dataset)
