import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Connect 元数据信息
class ConnectMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "connect"
        self.instance = 67557
        self.feats = 43
        self.in_features = 126
        self.out_features = 3
        self.batch_size = 64
        self.is_balanced = False
        self.class_ratio = [10650, 28439, 4147]
        self.data_dir = "/.data/connect"

        self.column_names = [f'pos_{i}' for i in range(42)] + ['class']
        self.label_name = 'class'
        self.multiclass_features = [f'pos_{i}' for i in range(42)]

class ConnectProcessor:
    def __init__(self):
        super().__init__()
        self.meta = ConnectMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        # 多类别型 one-hot
        df_multiclass_encoded = pd.get_dummies(df[self.meta.multiclass_features])
        # encode label
        df[self.meta.label_name] = df[self.meta.label_name].map({'loss': 0, 'win': 1, 'draw': 2})
        # 合并
        X = pd.concat([df_multiclass_encoded], axis=1)
        y = df[self.meta.label_name]
        feature_names = X.columns.tolist()

        # 构造特征类型映射字典
        feature_types = {}
        for col in df_multiclass_encoded.columns:
            feature_types[col] = "multiclass"

        feature_indices = {
            "numerical": [],
            "binary": [],
            "multiclass": []
        }

        for idx, col in enumerate(feature_names):
            feature_indices["multiclass"].append(idx)

        return y.values, X.astype(np.float32), feature_names, feature_types, feature_indices

    @staticmethod
    def fix_columns(test_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
        for col in set(train_columns) - set(test_df.columns):
            test_df[col] = 0
        return test_df[train_columns]

class ConnectDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = ConnectProcessor()

        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = [f'pos_{i}' for i in range(42)] + ['class']

        # 加载数据
        df_train = pd.read_csv(train_path, header=None, names=self.column_names, skipinitialspace=True)
        df_test = pd.read_csv(test_path, header=None, names=self.column_names, skipinitialspace=True)

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

























