import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader

class BankMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "bank"
        self.instance = 45211
        self.feats = 21
        self.in_features = 54
        self.out_features = 2
        self.batch_size = 64
        self.is_balanced = False
        self.class_ratio = [23411, 2949]
        self.data_dir = "/.data/bank"
        self.column_names = [
            "age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
            "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed", "class"
        ]
        # 标签名称
        self.label_name = 'class'
        self.continuous_features = [
            "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"
        ]
        self.binary_features = [
            "default", "housing", "loan"
        ]

        self.multiclass_features = [
            "job", "marital", "education", "contact", "month", "day_of_week", "poutcome"
        ]

        self.modes = {
            'marital': 'married',
            'default': 'no',
            'housing': 'yes',
            'loan': 'no',
            'job': 'admin.',
            'education': 'university.degree',
            'contact': 'cellular',
            'month': 'may',
            'day_of_week': 'thu',
            'poutcome': 'nonexistent'
        }

        self.means = {
            'age': 40.02406040594348,
            'duration': 258.2850101971448,
            'campaign': 2.567592502670681,
            'pdays': 962.4754540157328,
            'previous': 0.17296299893172767,
            'emp.var.rate': 0.08188550063125165,
            'cons.price.idx': 93.57566436826262,
            'cons.conf.idx': -40.50260027192386,
            'euribor3m': 3.621290812858114,
            'nr.employed': 5167.035910944936
        }
        self.stds = {
            'age': 10.421249980934235,
            'duration': 259.2792488364648,
            'campaign': 2.7700135429021127,
            'pdays': 186.9109073447436,
            'previous': 0.4949010798393183,
            'emp.var.rate': 1.5709597405172309,
            'cons.price.idx': 0.5788400489541813,
            'cons.conf.idx': 4.628197856174375,
            'euribor3m': 1.7344474048511707,
            'nr.employed': 72.25152766826527
        }

class BankProcessor:
    def __init__(self):
        super().__init__()
        self.meta = BankMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        # 填充缺失值
        df.replace("unknown", pd.NaT, inplace=True)
        df.fillna(self.meta.modes, inplace=True)
        # 二值编码
        df['default'] = df['default'].map({'yes': 1, 'no': 0})
        df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
        df['loan'] = df['loan'].map({'yes': 1, 'no': 0})

        # 数值型
        df[self.meta.continuous_features] = df[self.meta.continuous_features].astype(float)
        df[self.meta.continuous_features] = df[self.meta.continuous_features].apply(
            lambda col: (col - self.meta.means[col.name]) / (self.meta.stds[col.name] + 1e-6)
        )
        # 多类别型 one-hot
        df_multiclass_encoded = pd.get_dummies(df[self.meta.multiclass_features])
        # encode label
        df['class'] = df['class'].map({'yes': 1, 'no': 0})
        # 合并
        X = pd.concat([df[self.meta.continuous_features], df[self.meta.binary_features], df_multiclass_encoded], axis=1)
        y = df[self.meta.label_name]
        feature_names = X.columns.tolist()
        # 构造特征类型映射字典
        feature_types = {}
        for col in self.meta.continuous_features:
            feature_types[col] = "numerical"
        for col in self.meta.binary_features:
            feature_types[col] = "binary"
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
            elif col in self.meta.binary_features:
                feature_indices["binary"].append(idx)
            else:
                feature_indices["multiclass"].append(idx)

        return y.values, X.astype(np.float32), feature_names, feature_types, feature_indices

    @staticmethod
    def fix_columns(test_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
        for col in set(train_columns) - set(test_df.columns):
            test_df[col] = 0
        return test_df[train_columns]

class BankDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train", ):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = BankProcessor()

        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = [
            "age", "job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "duration", "campaign", "pdays",
            "previous", "poutcome", "emp.var.rate", "cons.price.idx",
            "cons.conf.idx", "euribor3m", "nr.employed", "class"
        ]
        # 加载数据
        df_train = pd.read_csv(self.train_path, header=None, names=self.column_names, skipinitialspace=True)
        df_test = pd.read_csv(self.test_path, header=None, names=self.column_names, skipinitialspace=True)
        # 预处理

        train_y, train_X, feature_names, feature_types, feature_indices = self.processor.process(df_train)
        test_y, test_X, test_col_name, test_col_type, test_indices = self.processor.process(df_test)
        self.feature_types = feature_types
        self.feature_indices = feature_indices
        # print(feature_names)
        # print(test_col_name)
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

def get_bank_dataset(data_dir):
    train_set = BankDataset(mode="train", data_dir=data_dir)
    val_set = BankDataset(mode="val", data_dir=data_dir)
    test_set = BankDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_bank_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = BankDataset(mode="train", data_dir=data_dir)
    val_set = BankDataset(mode="val", data_dir=data_dir)
    test_set = BankDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_bank_dataloader(data_dir, batch_size):
    train_set = BankDataset(mode="train", data_dir=data_dir)
    val_set = BankDataset(mode="val", data_dir=data_dir)
    test_set = BankDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_bank_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = BankDataset(mode="train", data_dir=data_dir)
    val_set = BankDataset(mode="val", data_dir=data_dir)
    test_set = BankDataset(mode="test", data_dir=data_dir)
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
    data_dir = "/data/ruipeng/workdir/autoreg/.data/bank"
    sample_ratio = 0.2
    batch_size = 64

    # 原始数据
    train_set_full = BankDataset(mode="train", data_dir=data_dir)

    # 采样后数据
    train_loader, val_loader, test_loader = get_bank_dataloader_sampled(
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