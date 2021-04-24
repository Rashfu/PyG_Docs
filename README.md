# PyG_Docs

图神经网络库PyG官方文档Introduction by Example部分的代码



## 环境配置

1. 首先安装Anaconda管理自己的虚拟环境
2. 配置pytorch，跟着[官网教程](https://pytorch.org/)安装即可（可自行搜索anaconda如何使用清华源）
3. 安装Pytorch_Geometric库，跟这[官方docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)安装即可（官网教程非常详细）

## 数据库dataset

所有的脚本第一次执行都会下载相关GNN数据集，并存放在上级目录datasets/下

目录结构

```目录
.
|-- PyG_Docs
`-- datasets
    |-- Cora
    |   `-- Cora
    |       |-- processed
    |       `-- raw
    |-- ENZYMES
    |   `-- ENZYMES
    |       |-- processed
    |       `-- raw
    `-- ShapeNet
        |-- processed
        `-- raw
```



## 待解决的问题

- batch内部详细执行过程
- [pytorch scatter docs](https://pytorch-scatter.readthedocs.io/en/latest/index.html)的阅读
- GCN高级解释的[博客](http://tkipf.github.io/graph-convolutional-networks/)