import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader


# Clickpred 元数据信息
class ClickpredMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "clickpred"
        self.instance = 399482
        self.feats = 12
        self.in_features = 15
        self.out_features = 2
        self.batch_size = 128
        self.is_balanced = False
        self.class_ratio = [212748, 42920]
        self.data_dir = "/.data/clickpred"

        self.column_names = [
            'impression', 'url_hash', 'ad_id', 'advertiser_id', 'depth', 'position',
            'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id', 'class'
        ]
        # 标签名称
        self.label_name = 'class'
        self.continuous_features = [
            'impression', 'url_hash', 'ad_id', 'advertiser_id',
            'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id'
        ]
        self.multiclass_features = ['depth', 'position']
        self.means = {
            'impression': 1.8794739187247484, 'url_hash': 9.647646398857994e+18, 'ad_id': 15975651.362314196,
            'advertiser_id': 22445.6497213892, 'query_id': 3184982.6906719203, 'keyword_id': 35086.02429896716,
            'title_id': 169952.42937103548, 'description_id': 108735.14388132632,
            'user_id': 3690651.1558142793
        }
        self.stds = {
            'impression': 34.03369354828914, 'url_hash': 4.982470656197324e+18, 'ad_id': 7227142.98379582,
            'advertiser_id': 11779.749444910238,
            'query_id': 5890907.276163105, 'keyword_id': 100889.0833404659, 'title_id': 457599.85022465914,
            'description_id': 321535.6168885692, 'user_id': 5502384.776849781
        }

class ClickpredProcessor:
    def __init__(self):
        super().__init__()
        self.meta = ClickpredMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        # 没有缺失项
        # 数值型
        df[self.meta.continuous_features] = df[self.meta.continuous_features].astype(float)
        df[self.meta.continuous_features] = df[self.meta.continuous_features].apply(
            lambda col: (col - self.meta.means[col.name]) / (self.meta.stds[col.name] + 1e-6)
        )
        df[self.meta.multiclass_features] = df[self.meta.multiclass_features].astype(str)
        df_multiclass_encoded = pd.get_dummies(df[self.meta.multiclass_features])
        # 合并
        X = pd.concat([df[self.meta.continuous_features], df_multiclass_encoded], axis=1)
        y = df[self.meta.label_name]
        feature_names = X.columns.tolist()
        # 构造特征类型映射字典
        feature_types = {}
        if not self.meta.continuous_features:
            for col in self.meta.continuous_features:
                feature_types[col] = "numerical"

        if not self.meta.multiclass_features:
            for col in df_multiclass_encoded.columns:
                feature_types[col] = "multiclass"

        feature_indices = {
            "numerical": [],
            "binary": [],
            "multiclass": []
        }

        for idx, col in enumerate(feature_names):
            if col in self.meta.continuous_features:
                feature_indices["numerical"].append(idx)
            else:
                feature_indices["multiclass"].append(idx)
        return y.values, X.astype(np.float32), feature_names, feature_types, feature_indices

    @staticmethod
    def fix_columns(test_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
        for col in set(train_columns) - set(test_df.columns):
            test_df[col] = 0
        return test_df[train_columns]

class ClickpredDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = ClickpredProcessor()

        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = [
            'impression', 'url_hash', 'ad_id', 'advertiser_id', 'depth', 'position',
            'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id', 'class'
        ]
        # 加载数据
        df_train = pd.read_csv(self.train_path, header=None, names=self.column_names, skipinitialspace=True)
        df_test = pd.read_csv(self.test_path, header=None, names=self.column_names, skipinitialspace=True)
        # 预处理
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

def get_clickpred_dataset(data_dir):
    train_set = ClickpredDataset(mode="train", data_dir=data_dir)
    val_set = ClickpredDataset(mode="val", data_dir=data_dir)
    test_set = ClickpredDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_clickpred_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = ClickpredDataset(mode="train", data_dir=data_dir)
    val_set = ClickpredDataset(mode="val", data_dir=data_dir)
    test_set = ClickpredDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_clickpred_dataloader(data_dir, batch_size):
    train_set = ClickpredDataset(mode="train", data_dir=data_dir)
    val_set = ClickpredDataset(mode="val", data_dir=data_dir)
    test_set = ClickpredDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_clickpred_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = ClickpredDataset(mode="train", data_dir=data_dir)
    val_set = ClickpredDataset(mode="val", data_dir=data_dir)
    test_set = ClickpredDataset(mode="test", data_dir=data_dir)
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
    data_dir = "/data/ruipeng/workdir/autoreg/.data/clickpred"
    sample_ratio = 0.2
    batch_size = 64

    # 原始数据
    train_set_full = ClickpredDataset(mode="train", data_dir=data_dir)

    # 采样后数据
    train_loader, val_loader, test_loader = get_clickpred_dataloader_sampled(
        data_dir=data_dir,
        batch_size=batch_size,
        sample_ratio=sample_ratio
    )
    train_set_sampled = train_loader.dataset

    # ====== 打印采样前类别分布 ======
    y_full = train_set_full.y.numpy()
    full_cls0 = (y_full == 0).sum()
    full_cls1 = (y_full == 1).sum()

    print("====== 采样前类别数量 ======")
    print(f"class 0: {full_cls0}")
    print(f"class 1: {full_cls1}")

    # ====== 打印采样后类别分布 ======
    y_sampled = train_set_sampled.y.numpy()
    sampled_cls0 = (y_sampled == 0).sum()
    sampled_cls1 = (y_sampled == 1).sum()

    print("\n====== 采样后类别数量 ======")
    print(f"class 0: {sampled_cls0}")
    print(f"class 1: {sampled_cls1}")

    # ====== 理论值 ======
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
        print("y:", y[:20])
        break