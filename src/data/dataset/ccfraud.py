import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader


# Ccfraud 元数据信息
class CcfraudMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "ccfraud"
        self.instance = 284807
        self.feats = 31
        self.in_features = 30
        self.out_features = 2
        self.batch_size = 128
        self.is_balanced = False
        self.class_ratio = [181965, 311]
        self.data_dir = "/.data/adult"
        self.column_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'] + ['class']
        # 标签名称
        self.label_name = 'class'
        self.continuous_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        self.means = {
            'Time': 94813.85957508067, 'V1': 1.1707700128069177e-15, 'V2': 3.384974329561085e-16,
            'V3': -1.3731499638785534e-15, 'V4': 2.0872677793956265e-15, 'V5': 9.604066317127324e-16,
            'V6': 1.4964939577269467e-15, 'V7': -5.572434155739711e-16, 'V8': 1.223460650955746e-16,
            'V9': -2.4065301356411542e-15, 'V10': 2.2388531569487897e-15, 'V11': 1.673326932726423e-15,
            'V12': -1.2549951995448174e-15, 'V13': 8.185510594995518e-16, 'V14': 1.204699590402754e-15,
            'V15': 4.887455859804944e-15, 'V16': 1.4359196824308506e-15, 'V17': -3.765185184385016e-16,
            'V18': 9.564149167014576e-16, 'V19': 1.039891656874743e-15, 'V20': 6.449613529467363e-16,
            'V21': 1.652570014667794e-16, 'V22': -3.4488417697414826e-16, 'V23': 2.640519479958323e-16,
            'V24': 4.472018120006513e-15, 'V25': 5.085444924364178e-16, 'V26': 1.6870702827691485e-15,
            'V27': -3.680735463677732e-16, 'V28': -1.2466624944587811e-16, 'Amount': 88.34961925093133
        }
        self.stds = {
            'Time': 47488.14595456582, 'V1': 1.9586958038574793, 'V2': 1.6513085794769742, 'V3': 1.5162550051777683,
            'V4': 1.4158685749409237, 'V5': 1.3802467340314384, 'V6': 1.3322710897575674, 'V7': 1.2370935981826603,
            'V8': 1.194352902669203, 'V9': 1.0986320892243226, 'V10': 1.0888497654025178, 'V11': 1.0207130277115524,
            'V12': 0.9992013895301388, 'V13': 0.9952742301251489, 'V14': 0.9585956112570686, 'V15': 0.9153160116104295,
            'V16': 0.876252887388374, 'V17': 0.8493370636743797, 'V18': 0.8381762095288368, 'V19': 0.8140405007685731,
            'V20': 0.770925024887114, 'V21': 0.7345240143713043, 'V22': 0.7257015604409169, 'V23': 0.6244602955949906,
            'V24': 0.605647067827154, 'V25': 0.521278070540938, 'V26': 0.482227013261055, 'V27': 0.4036324949650267,
            'V28': 0.33008326416025413, 'Amount': 250.1201092402221
        }

class CcfraudProcessor:
    def __init__(self):
        super().__init__()
        self.meta = CcfraudMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        # 没有缺失项
        # 数值型
        df[self.meta.continuous_features] = df[self.meta.continuous_features].astype(float)
        df[self.meta.continuous_features] = df[self.meta.continuous_features].apply(
            lambda col: (col - self.meta.means[col.name]) / (self.meta.stds[col.name] + 1e-6)
        )
        # 合并
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

class CcfraudDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = CcfraudProcessor()

        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'] + ['class']
        # 加载数据
        df_train = pd.read_csv(train_path, header=None, names=self.column_names, skipinitialspace=True)
        df_test = pd.read_csv(test_path, header=None, names=self.column_names, skipinitialspace=True)
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

def get_ccfraud_dataset(data_dir):
    train_set = CcfraudDataset(mode="train", data_dir=data_dir)
    val_set = CcfraudDataset(mode="val", data_dir=data_dir)
    test_set = CcfraudDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_ccfraud_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = CcfraudDataset(mode="train", data_dir=data_dir)
    val_set = CcfraudDataset(mode="val", data_dir=data_dir)
    test_set = CcfraudDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set


def get_ccfraud_dataloader(data_dir, batch_size):
    train_set = CcfraudDataset(mode="train", data_dir=data_dir)
    val_set = CcfraudDataset(mode="val", data_dir=data_dir)
    test_set = CcfraudDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_ccfraud_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = CcfraudDataset(mode="train", data_dir=data_dir)
    val_set = CcfraudDataset(mode="val", data_dir=data_dir)
    test_set = CcfraudDataset(mode="test", data_dir=data_dir)
    # 分层按比例采样（推荐）
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # train_d, val_d, test_d = get_adult_dataloader(data_dir="/home/zrp/pycharmProjects/autoreg/.data/adult", batch_size=64)
    # print(len(train_d))
    # print(len(val_d))
    # print(len(test_d))
    print("----")
    # for x,y in val_d:
    #     print(x, y)
    #     break
    # print("----")
    # for x, y in val_d:
    #     print(x, y)
    data_dir = "/data/ruipeng/workdir/autoreg/.data/ccfraud"
    sample_ratio = 0.2
    batch_size = 128

    # 原始数据
    train_set_full = CcfraudDataset(mode="train", data_dir=data_dir)

    # 采样后数据
    train_loader, val_loader, test_loader = get_ccfraud_dataloader_sampled(
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