import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Criteo 元数据信息
class CriteoMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "criteo"
        self.instance =  45840617
        self.feats = 40
        self.in_features = 39
        self.out_features = 2
        self.batch_size = 4096
        self.is_balanced = False
        self.class_ratio =  [34095179, 11745438]
        self.data_dir = "/.data/criteo"
        self.column_names = [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)] + [ 'label' ]
        self.label_name = 'label'

        self.continuous_features = [f'I{i}' for i in range(1, 14)]
        self.multiclass_features =  [f'C{i}' for i in range(1, 27)]
        self.binary_features = []

class CriteoProcessor:
    def __init__(self):
        super().__init__()
        self.meta = CriteoMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        X = pd.concat([df[self.meta.continuous_features], df[self.meta.multiclass_features]], axis=1)
        y = df[self.meta.label_name]
        # 特征名称列表
        feature_names = X.columns.tolist()
        # 构造特征类型映射字典
        feature_types = {}

        for col in self.meta.continuous_features:
            feature_types[col] = "numerical"
        for col in self.meta.multiclass_features:
            feature_types[col] = "multiclass"

        # 构造索引映射
        feature_indices = {
            "numerical": [],
            "binary": [],
            "multiclass": []
        }
        for idx, col in enumerate(feature_names):
            if col in self.meta.continuous_features:
                feature_indices["numerical"].append(idx)
            elif col in self.meta.binary_features:
                feature_indices["binary"].append(idx)
            else:
                feature_indices["multiclass"].append(idx)
        return y.values, X.astype(np.float32), feature_names, feature_types, feature_indices


class CriteoDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = CriteoProcessor()
        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)] + [ 'label' ]
        # 加载数据
        if self.mode == "train":
            df_train = pd.read_csv(self.train_path, header=None, names=self.column_names, skipinitialspace=True, low_memory=False)
            # 预处理
            train_y, train_X, feature_names, feature_types, feature_indices = self.processor.process(df=df_train)
            self.feature_types = feature_types
            self.feature_indices = feature_indices
            train_X = torch.FloatTensor(train_X.values)
            train_y = torch.LongTensor(train_y)
            self.X = train_X
            self.y = train_y

        if self.mode == "test" or self.mode == "val":
            df_test = pd.read_csv(self.test_path, header=None, names=self.column_names, skipinitialspace=True, low_memory=False)
            test_y, test_X, test_col_name, test_col_type, test_indices = self.processor.process(df=df_test)
            self.feature_types = test_col_type
            self.feature_indices = test_indices
            test_X = torch.FloatTensor(test_X.values)
            test_y = torch.LongTensor(test_y)
            self.X = test_X
            self.y = test_y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

def sample_balanced(dataset, ratio):
    X, y = dataset.X, dataset.y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=42)
    for train_idx, _ in sss.split(X, y):
        # 只保留按类别采样后的 X 和 y
        dataset.X = X[train_idx]
        dataset.y = y[train_idx]
        break

    return dataset

def get_criteo_dataset(data_dir):
    train_set = CriteoDataset(mode="train", data_dir=data_dir)
    val_set = CriteoDataset(mode="val", data_dir=data_dir)
    test_set = CriteoDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_criteo_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = CriteoDataset(mode="train", data_dir=data_dir)
    val_set = CriteoDataset(mode="val", data_dir=data_dir)
    test_set = CriteoDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_criteo_dataloader(data_dir, batch_size):
    train_set = CriteoDataset(mode="train", data_dir=data_dir)
    val_set = CriteoDataset(mode="val", data_dir=data_dir)
    test_set = CriteoDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_criteo_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = CriteoDataset(mode="train", data_dir=data_dir)
    val_set = CriteoDataset(mode="val", data_dir=data_dir)
    test_set = CriteoDataset(mode="test", data_dir=data_dir)
    # 分层按比例采样（推荐）
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # print(f"[Sampled Loader] train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    print("----")
    data_dir = "/data/ruipeng/workdir/autoreg/.data/criteo"
    sample_ratio = 0.2
    batch_size = 4096

    # # 原始数据
    train_set_full = CriteoDataset(mode="train", data_dir=data_dir)
    # ====== 打印采样前类别分布 ======
    y_full = train_set_full.y.numpy()
    full_cls0 = (y_full == 0).sum()
    full_cls1 = (y_full == 1).sum()
    print("====== 采样前类别数量 ======")
    print(f"class 0: {full_cls0}")
    print(f"class 1: {full_cls1}")

    # 采样后数据
    train_loader, val_loader, test_loader = get_criteo_dataloader_sampled(
        data_dir=data_dir,
        batch_size=batch_size,
        sample_ratio=sample_ratio
    )
    train_set_sampled = train_loader.dataset

    # ====== 打印采样后类别分布 ======
    y_sampled = train_set_sampled.y.numpy()
    sampled_cls0 = (y_sampled == 0).sum()
    sampled_cls1 = (y_sampled == 1).sum()

    print("\n====== 采样后类别数量 ======")
    print(f"class 0: {sampled_cls0}")
    print(f"class 1: {sampled_cls1}")

    #  ====== 理论值 ======
    print("\n====== 理论采样数量（按比例） ======")
    print(f"class 0 应为: {int(full_cls0 * sample_ratio)}")
    print(f"class 1 应为: {int(full_cls1 * sample_ratio)}")

    # ====== 数据集大小 ======
    print("\n====== 数据集大小 ======")
    print(f"采样前: {len(train_set_full)}")
    print(f"采样后: {len(train_set_sampled)}")

    # ====== 查看一个 batch（可选） ======
    print("\n====== 查看一个 batch ======")
    for X, y in train_loader:
        print("X shape:", X.shape)
        print("X:", X[:20])
        print("y:", y[:20])
        break