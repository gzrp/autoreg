import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Adult 元数据信息
class AdultMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "adult"
        self.instance = 48842
        self.feats = 15
        self.in_features = 106
        self.out_features = 2
        self.batch_size = 128
        self.is_balanced = False
        self.class_ratio = [29724, 9349],
        self.data_dir = "/.data/adult"

        self.column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'
        ]
        self.label_name = 'class'
        self.continuous_features = [
            "age", "education-num", "capital-gain", "capital-loss", "hours-per-week"
        ]
        self.multiclass_features = [
            "workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"
        ]
        self.binary_features = ["sex"]
        self.means = {
            'age': 38.64358543876172,
            'education-num': 10.078088530363212,
            'capital-gain': 1079.0676262233324,
            'capital-loss': 87.50231358257237,
            'hours-per-week': 40.422382375824085
        }
        self.stds = {
            'age': 13.71050993444322,
            'education-num': 2.5709727555918307,
            'capital-gain': 7452.019057653448,
            'capital-loss': 403.0045521244552,
            'hours-per-week': 12.39144402425593
        }
        self.modes = {
            'sex': ' Male',
            'workclass': ' Private',
            'education': ' HS-grad',
            'marital-status': ' Married-civ-spouse',
            'occupation': ' Prof-specialty',
            'relationship': ' Husband',
            'race': ' White',
            'native-country': ' United-States'
        }
        self.medians = {
            'age': 37.0,
            'education-num': 10.0,
            'capital-gain': 0.0,
            'capital-loss': 0.0,
            'hours-per-week': 40.0
        }

class AdultProcessor:
    def __init__(self):
        super().__init__()
        self.meta = AdultMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        df.drop(columns=['fnlwgt'], inplace=True)
        df.replace("?", pd.NaT, inplace=True)
        df.fillna(self.meta.modes, inplace=True)
        # 二值编码
        df['sex'] = df['sex'].map({"Male": 1, "Female": 0})
        # 预测标签编码
        df['class'] = df['class'].map({"<=50K": 0, ">50K": 1 })
        # 数值列归一化
        df[self.meta.continuous_features] = df[self.meta.continuous_features].apply(
            lambda col: (col - self.meta.means[col.name]) / (self.meta.stds[col.name] + 1e-6)
        )

        # 类别型 One-hot 编码
        df_multiclass_encoded = pd.get_dummies(df[self.meta.multiclass_features])

        # 合并处理后的列
        X = pd.concat([df[self.meta.continuous_features], df[self.meta.binary_features], df_multiclass_encoded], axis=1)
        y = df[self.meta.label_name]
        # 特征名称列表
        feature_names = X.columns.tolist()

        # 构造特征类型映射字典
        feature_types = {}
        if not self.meta.continuous_features:
            for col in self.meta.continuous_features:
                feature_types[col] = "numerical"
        if not self.meta.binary_features:
            for col in self.meta.binary_features:
                feature_types[col] = "binary"
        if not self.meta.multiclass_features:
            for col in df_multiclass_encoded.columns:
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

    @staticmethod
    def fix_columns(test_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
        for col in set(train_columns) - set(test_df.columns):
            test_df[col] = 0
        return test_df[train_columns]

class AdultDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = AdultProcessor()

        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                             'marital-status', 'occupation', 'relationship', 'race', 'sex',
                             'capital-gain', 'capital-loss', 'hours-per-week',
                             'native-country', 'class']
        # 加载数据
        df_train = pd.read_csv(self.train_path, header=None, names=self.column_names, skipinitialspace=True,
                               low_memory=False)

        df_test = pd.read_csv(self.test_path, header=None, names=self.column_names, skipinitialspace=True)

        # 预处理

        train_y, train_X, feature_names, feature_types, feature_indices = self.processor.process(df=df_train)

        test_y, test_X, test_col_name, test_col_type, test_indices = self.processor.process(df=df_test)
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

def get_adult_dataset(data_dir):
    train_set = AdultDataset(mode="train", data_dir=data_dir)
    val_set = AdultDataset(mode="val", data_dir=data_dir)
    test_set = AdultDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_adult_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = AdultDataset(mode="train", data_dir=data_dir)
    val_set = AdultDataset(mode="val", data_dir=data_dir)
    test_set = AdultDataset(mode="test", data_dir=data_dir)
    # 按比例采样
    def sample(dataset, ratio):
        n = int(len(dataset) * ratio)
        dataset.X = dataset.X[:n]
        dataset.y = dataset.y[:n]
        return dataset

    train_set = sample(train_set, sample_ratio)
    val_set = sample(val_set, sample_ratio)
    test_set = sample(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_adult_dataloader(data_dir, batch_size):
    start_t = time.time()
    train_set = AdultDataset(mode="train", data_dir=data_dir)
    val_set = AdultDataset(mode="val", data_dir=data_dir)
    test_set = AdultDataset(mode="test", data_dir=data_dir)
    start_t2 = time.time()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_adult_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = AdultDataset(mode="train", data_dir=data_dir)
    val_set = AdultDataset(mode="val", data_dir=data_dir)
    test_set = AdultDataset(mode="test", data_dir=data_dir)
    # 按比例采样
    def sample(dataset, ratio):
        n = int(len(dataset) * ratio)
        dataset.X = dataset.X[:n]
        dataset.y = dataset.y[:n]
        return dataset

    train_set = sample(train_set, sample_ratio)
    val_set = sample(val_set, sample_ratio)
    test_set = sample(test_set, sample_ratio)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # print(f"[Sampled Loader] train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_d, val_d, test_d = get_adult_dataloader(data_dir="/home/zrp/pycharmProjects/autoreg/.data/adult", batch_size=64)
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