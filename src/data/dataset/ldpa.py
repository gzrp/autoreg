import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
#
# "ldpa": {
#     "instance": 164860,
#     "feats": 8,
#     "in_features": 14,
#     "out_features": 11,
#     "batch_size": 128,
#     "is_balanced": False,
#     "class_ratio": [1916, 1827, 20914, 34931, 3997, 3358, 17350, 1100, 7519, 11705, 893],
#     "data_dir": "D:\\User\\zhangruipeng\\PycharmProjects\\new-ai-engine\\.data\\ldpa"
# },

# LDPA 元数据信息

class LdpaMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "ldpa"
        self.instance = 164860
        self.feats = 8
        self.in_features = 14
        self.out_features = 11
        self.batch_size = 128
        self.is_balanced = False
        self.class_ratio = [2378, 2278, 26168, 43584, 4935, 4168, 21795, 1365, 9423, 14689, 1105]
        self.data_dir = "/.data/ldpa"
        self.column_names = [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'class'
        ]
        self.label_name = 'class'
        self.continuous_features = [
            'V3', 'V4', 'V5', 'V6', 'V7'
        ]
        self.multiclass_features = ['V1', 'V2']
        self.means = {
            'V3': 82429.6849629989,
            'V4': 82416.60670872255,
            'V5': 81955.86375712726,
            'V6': 81869.99064054349,
            'V7': 82270.7156011161
        }
        self.stds = {
            'V3': 47590.865917142044,
            'V4': 47583.958209776145,
            'V5': 47231.43061579646,
            'V6': 47224.778309513946,
            'V7': 47463.95332903256
        }

class LdpaProcessor:
    def __init__(self):
        super().__init__()
        self.meta = LdpaMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        # 数值型
        df[self.meta.continuous_features] = df[self.meta.continuous_features].astype(float)
        df[self.meta.continuous_features] = df[self.meta.continuous_features].apply(
            lambda col: (col - self.meta.means[col.name]) / (self.meta.stds[col.name] + 1e-6)
        )
        # 多类别型 one-hot
        df[self.meta.multiclass_features] = df[self.meta.multiclass_features].astype(str)
        df_multiclass_encoded = pd.get_dummies(df[self.meta.multiclass_features])

        # encode label
        df[self.meta.label_name] = df[self.meta.label_name].astype(int) - 1
        # 合并
        X = pd.concat([df[self.meta.continuous_features], df_multiclass_encoded], axis=1)
        y = df[self.meta.label_name]

        feature_names = X.columns.tolist()
        # 构造特征类型映射字典
        feature_types = {}
        for col in self.meta.continuous_features:
            feature_types[col] = "numerical"
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

class LdpaDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = LdpaProcessor()
        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'class'
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
            self.X = test_X
            self.y = test_y
            # self.X = train_X[int(0.8 * len(train_X)):]
            # self.y = train_y[int(0.8 * len(train_y)):]
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

def get_ldpa_dataset(data_dir):
    train_set = LdpaDataset(mode="train", data_dir=data_dir)
    val_set = LdpaDataset(mode="val", data_dir=data_dir)
    test_set = LdpaDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_ldpa_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = LdpaDataset(mode="train", data_dir=data_dir)
    val_set = LdpaDataset(mode="val", data_dir=data_dir)
    test_set = LdpaDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_ldpa_dataloader(data_dir, batch_size):
    train_set = LdpaDataset(mode="train", data_dir=data_dir)
    val_set = LdpaDataset(mode="val", data_dir=data_dir)
    test_set = LdpaDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_ldpa_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = LdpaDataset(mode="train", data_dir=data_dir)
    val_set = LdpaDataset(mode="val", data_dir=data_dir)
    test_set = LdpaDataset(mode="test", data_dir=data_dir)
    # 分层按比例采样（推荐）
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data_dir = "/data/ruipeng/workdir/autoreg/.data/ldpa"
    sample_ratio = 0.2
    batch_size = 128
    # 原始数据
    train_set_full = LdpaDataset(mode="train", data_dir=data_dir)

    # 采样后数据
    train_loader, val_loader, test_loader = get_ldpa_dataloader_sampled(
        data_dir=data_dir,
        batch_size=batch_size,
        sample_ratio=sample_ratio
    )
    train_set_sampled = train_loader.dataset
    # ====== 打印采样前类别分布 ======
    y_full = train_set_full.y.numpy()
    full_cls0 = (y_full == 0).sum()
    full_cls1 = (y_full == 1).sum()
    full_cls2 = (y_full == 2).sum()
    full_cls3 = (y_full == 3).sum()
    full_cls4 = (y_full == 4).sum()
    full_cls5 = (y_full == 5).sum()
    full_cls6 = (y_full == 6).sum()
    full_cls7 = (y_full == 7).sum()
    full_cls8 = (y_full == 8).sum()
    full_cls9 = (y_full == 9).sum()
    full_cls10 = (y_full == 10).sum()

    print("====== 采样前类别数量 ======")
    print(f"class 0: {full_cls0}")
    print(f"class 1: {full_cls1}")
    print(f"class 2: {full_cls2}")
    print(f"class 3: {full_cls3}")
    print(f"class 4: {full_cls4}")
    print(f"class 5: {full_cls5}")
    print(f"class 6: {full_cls6}")
    print(f"class 7: {full_cls7}")
    print(f"class 8: {full_cls8}")
    print(f"class 9: {full_cls9}")
    print(f"class 10: {full_cls10}")
    # ====== 打印采样后类别分布 ======
    y_sampled = train_set_sampled.y.numpy()
    sampled_cls0 = (y_sampled == 0).sum()
    sampled_cls1 = (y_sampled == 1).sum()
    sampled_cls2 = (y_sampled == 2).sum()
    sampled_cls3 = (y_sampled == 3).sum()
    sampled_cls4 = (y_sampled == 4).sum()
    sampled_cls5 = (y_sampled == 5).sum()
    sampled_cls6 = (y_sampled == 6).sum()
    sampled_cls7 = (y_sampled == 7).sum()
    sampled_cls8 = (y_sampled == 8).sum()
    sampled_cls9 = (y_sampled == 9).sum()
    sampled_cls10 = (y_sampled == 10).sum()

    print("\n====== 采样后类别数量 ======")
    print(f"class 0: {sampled_cls0}")
    print(f"class 1: {sampled_cls1}")
    print(f"class 2: {sampled_cls2}")
    print(f"class 3: {sampled_cls3}")
    print(f"class 4: {sampled_cls4}")
    print(f"class 5: {sampled_cls5}")
    print(f"class 6: {sampled_cls6}")
    print(f"class 7: {sampled_cls7}")
    print(f"class 8: {sampled_cls8}")
    print(f"class 9: {sampled_cls9}")
    print(f"class 10: {sampled_cls10}")


    # ====== 理论值 ======
    print("\n====== 理论采样数量（按比例） ======")
    print(f"class 0 应为: {int(full_cls0 * sample_ratio)}")
    print(f"class 1 应为: {int(full_cls1 * sample_ratio)}")
    print(f"class 2 应为: {int(full_cls2 * sample_ratio)}")
    print(f"class 3 应为: {int(full_cls3 * sample_ratio)}")

    print(f"class 4 应为: {int(full_cls4 * sample_ratio)}")
    print(f"class 5 应为: {int(full_cls5 * sample_ratio)}")

    print(f"class 6 应为: {int(full_cls6 * sample_ratio)}")
    print(f"class 7 应为: {int(full_cls7 * sample_ratio)}")

    print(f"class 8 应为: {int(full_cls8 * sample_ratio)}")
    print(f"class 9 应为: {int(full_cls9 * sample_ratio)}")
    print(f"class 10 应为: {int(full_cls10 * sample_ratio)}")

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