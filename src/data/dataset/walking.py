import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader

# Walking 元数据信息
class WalkingMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "walking"
        self.instance = 149332
        self.feats = 5
        self.in_features = 4
        self.out_features = 22
        self.batch_size = 128
        self.is_balanced = False
        self.class_ratio = [3241, 2451, 751, 4461, 728, 3116, 2407, 2206, 5126, 2007, 3627, 3103, 4306, 7662, 2316, 1134,
                        14131, 13212, 578, 10845, 2009, 6155]
        self.data_dir = "/.data/walking"

        self.column_names = ['timestep', 'x', 'y', 'z', 'class']
        self.label_name = 'class'
        self.continuous_features = ['timestep', 'x', 'y', 'z']

        self.means = {
            'timestep': 185.5599163981062,
            'x': -1.6553281144630756,
            'y': 8.769386881572606,
            'z': 0.5555795400985721
        }
        self.stds = {
            'timestep': 167.16082969335665,
            'x': 2.8669720380932247,
            'y': 2.7722346309853347,
            'z': 3.147621164777253
        }

class WalkingProcessor:
    def __init__(self):
        super().__init__()
        self.meta = WalkingMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        # 没有缺失项 没有二值 没有多值
        # 数值型
        df[self.meta.continuous_features] = df[self.meta.continuous_features].astype(float)
        df[self.meta.continuous_features] = df[self.meta.continuous_features].apply(
            lambda col: (col - self.meta.means[col.name]) / (self.meta.stds[col.name] + 1e-6)
        )
        X = pd.concat([df[self.meta.continuous_features]], axis=1)
        y = df[self.meta.label_name]
        feature_names = X.columns.tolist()
        # 构造特征类型映射字典
        feature_types = {}
        for col in self.meta.continuous_features:
            feature_types[col] = "numerical"

        feature_indices = {
            "numerical": [],
            "binary": [],
            "multiclass": []
        }

        for idx, col in enumerate(feature_names):
            if col in self.meta.continuous_features:
                feature_indices["numerical"].append(idx)

        return y.values, X.astype(np.float32), feature_names, feature_types, feature_indices

    @staticmethod
    def fix_columns(test_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
        for col in set(train_columns) - set(test_df.columns):
            test_df[col] = 0
        return test_df[train_columns]

class WalkingDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = WalkingProcessor()

        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = ['timestep', 'x', 'y', 'z', 'class']
        # 加载数据
        df_train = pd.read_csv(self.train_path, header=None, names=self.column_names, skipinitialspace=True)
        df_test = pd.read_csv(self.test_path, header=None, names=self.column_names, skipinitialspace=True)
        # 预处理处理
        train_y, train_X, feature_names, feature_types, feature_indices = self.processor.process(df_train)
        test_y, test_X, test_col_name, test_col_type, test_indices = self.processor.process(df_test)
        self.feature_types = feature_types
        self.feature_indices = feature_indices
        # 对齐列
        test_X = self.processor.fix_columns(test_X, train_X.columns)
        # 转换为 torch.Tensor
        train_X, test_X = torch.FloatTensor(train_X.values), torch.FloatTensor(test_X.values)
        train_y, test_y = torch.LongTensor(train_y), torch.LongTensor(test_y)

        # 划分 train / val
        if mode == "train":
            self.X = train_X
            self.y = train_y
            # self.X = train_X[:int(0.8 * len(train_X))]
            # self.y = train_y[:int(0.8 * len(train_y))]
        elif mode == "val":
            # self.X = train_X[int(0.8 * len(train_X)):]
            # self.y = train_y[int(0.8 * len(train_y)):]
            self.X = test_X
            self.y = test_y
        elif mode == "test":
            self.X = test_X
            self.y = test_y
        else:
            raise ValueError(f"Unknown mode: {mode}")

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

def get_walking_dataset(data_dir):
    train_set = WalkingDataset(mode="train", data_dir=data_dir)
    val_set = WalkingDataset(mode="val", data_dir=data_dir)
    test_set = WalkingDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_walking_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = WalkingDataset(mode="train", data_dir=data_dir)
    val_set = WalkingDataset(mode="val", data_dir=data_dir)
    test_set = WalkingDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_walking_dataloader(data_dir, batch_size):
    train_set = WalkingDataset(mode="train", data_dir=data_dir)
    val_set = WalkingDataset(mode="val", data_dir=data_dir)
    test_set = WalkingDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_walking_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = WalkingDataset(mode="train", data_dir=data_dir)
    val_set = WalkingDataset(mode="val", data_dir=data_dir)
    test_set = WalkingDataset(mode="test", data_dir=data_dir)
    # 分层按比例采样（推荐）
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    print("---- WALKING Sampling Test ----")

    data_dir = "/data/ruipeng/workdir/autoreg/.data/walking"  # ⚠️改成你的路径
    sample_ratio = 0.2
    batch_size = 128

    # ------------ 原始训练集 ------------
    train_set_full = WalkingDataset(mode="train", data_dir=data_dir)

    # ------------ 采样后数据 (dataset) ------------
    train_set_sampled, _, _ = get_walking_dataset_sampled(
        data_dir=data_dir,
        sample_ratio=sample_ratio
    )

    # ------------ 采样后 dataloader (注意 dataset 来自 loader.dataset) ------------
    train_loader, val_loader, test_loader = get_walking_dataloader_sampled(
        data_dir=data_dir,
        batch_size=batch_size,
        sample_ratio=sample_ratio
    )
    train_set_sampled_loader = train_loader.dataset

    # ================================================================
    #                打印采样前类别分布
    # ================================================================
    print("\n====== 采样前类别数量 ======")
    import collections

    y_full = train_set_full.y.numpy()
    full_counter = collections.Counter(y_full)
    for cls, num in sorted(full_counter.items()):
        print(f"class {cls}: {num}")

    # ================================================================
    #                打印采样后类别分布
    # ================================================================
    print("\n====== 采样后类别数量 ======")
    y_sampled = train_set_sampled.y.numpy()
    sampled_counter = collections.Counter(y_sampled)
    for cls, num in sorted(sampled_counter.items()):
        print(f"class {cls}: {num}")

    # ================================================================
    #                理论采样值
    # ================================================================
    print("\n====== 理论采样数量（按 sample_ratio） ======")
    for cls, num in sorted(full_counter.items()):
        print(f"class {cls}: {int(num * sample_ratio)}")

    # ================================================================
    #                数据集大小
    # ================================================================
    print("\n====== 数据集大小 ======")
    print(f"采样前: {len(train_set_full)}")
    print(f"采样后: {len(train_set_sampled)}")

    # ================================================================
    #                查看一个 batch（可选）
    # ================================================================
    print("\n====== 查看一个 batch ======")
    for X, y in train_loader:
        print("X shape:", X.shape)
        print("y:", y[:20])
        break