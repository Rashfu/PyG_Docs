from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import KNNGraph

# # --------读取的是点云数据------------
# point_dataset = ShapeNet(root='../datasets/ShapeNet', categories=['Airplane'])
# print(point_dataset)
#
# point_data = point_dataset[0]
# print(point_data)
#
# print('\n')

# --------利用transform转换成图数据-----------
graph_dataset = ShapeNet(root='../datasets/ShapeNet', categories=['Airplane'],
                         pre_transform=KNNGraph(k=6))
print(graph_dataset)

graph_data = graph_dataset[0]
print(graph_data)

# Tips: 两个数据集读取，都会在数据集文件夹下生成processed文件，所以每次换读取方式记得删除文件
