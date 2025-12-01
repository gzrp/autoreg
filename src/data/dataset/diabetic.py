import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Diabetic 元数据信息
class DiabeticMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "diabetic"
        self.instance = 101766
        self.feats = 50
        self.in_features = 2347
        self.out_features = 2
        self.batch_size = 64
        self.is_balanced = False
        self.class_ratio = [90409, 11357]
        self.data_dir = "/.data/diabetic"

        self.column_names = [
            'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures',	'num_procedures',
            'num_medications', 'number_outpatient',	'number_emergency',	'number_inpatient',	'diag_1', 'diag_2',	'diag_3',
            'number_diagnoses',	'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
            'acarbose',	'miglitol',	'troglitazone',	'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
            'diabetesMed', 'readmitted'
        ]
        self.label_name = 'readmitted'
        self.multiclass_features = [
            'race', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'time_in_hospital', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
            'max_glu_serum', 'A1Cresult', 'metformin','repaglinide', 'nateglinide',	'chlorpropamide', 'glimepiride',
            'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'tolazamide', 'insulin',
            'glyburide-metformin'
        ]
        self.binary_features = [
            'gender', 'acetohexamide', 'tolbutamide', 'troglitazone', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
        ]

class DiabeticProcessor:
    def __init__(self):
        super().__init__()
        self.meta = DiabeticMetaData()

    def process(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        df.drop(columns=['encounter_id', 'patient_nbr', 'payer_code', 'examide', 'citoglipton'], inplace=True)
        df.replace("?", "Unknown", inplace=True)
        # 二值编码
        df['gender'] = df['gender'].map({"Male": 1, "Female": 0})
        df['acetohexamide'] = df['acetohexamide'].map({"Steady": 1, "No": 0})
        df['tolbutamide'] = df['tolbutamide'].map({"Steady": 1, "No": 0})
        df['troglitazone'] = df['troglitazone'].map({"Steady": 1, "No": 0})
        df['glipizide-metformin'] = df['glipizide-metformin'].map({"Steady": 1, "No": 0})
        df['glimepiride-pioglitazone'] = df['glimepiride-pioglitazone'].map({"Steady": 1, "No": 0})
        df['metformin-rosiglitazone'] = df['metformin-rosiglitazone'].map({"Steady": 1, "No": 0})
        df['metformin-pioglitazone'] = df['metformin-pioglitazone'].map({"Steady": 1, "No": 0})
        df['change'] = df['change'].map({"Ch": 1, "No": 0})
        df['diabetesMed'] = df['diabetesMed'].map({"Yes": 1, "No": 0})
        # 预测标签编码
        df['readmitted'] = df['readmitted'].map({"<30": 1, ">30": 0, 'NO': 0 })

        # 类别型 One-hot 编码
        df_multiclass_encoded = pd.get_dummies(df[self.meta.multiclass_features])

        # 合并处理后的列
        X = pd.concat([df[self.meta.binary_features], df_multiclass_encoded], axis=1)
        y = df[self.meta.label_name]
        # 特征名称列表
        feature_names = X.columns.tolist()

        # 构造特征类型映射字典
        feature_types = {}

        for col in self.meta.binary_features:
            feature_types[col] = "binary"
        for col in df_multiclass_encoded.columns:
            feature_types[col] = "multiclass"

        # 构造索引映射
        feature_indices = {
            "numerical": [],
            "binary": [],
            "multiclass": []
        }
        for idx, col in enumerate(feature_names):
            if col in self.meta.binary_features:
                feature_indices["binary"].append(idx)
            else:
                feature_indices["multiclass"].append(idx)
        return y.values, X.astype(np.float32), feature_names, feature_types, feature_indices

    @staticmethod
    def fix_columns(test_df: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
        missing = list(set(train_columns) - set(test_df.columns))
        # 一次性添加DataFrame（不会碎片化）
        add_df = pd.DataFrame(0, index=test_df.index, columns=missing)
        test_df = pd.concat([test_df, add_df], axis=1)

        # 按训练顺序重新排序列
        return test_df[train_columns]

class DiabeticDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = DiabeticProcessor()

        self.train_path = os.path.join(data_dir, 'train.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = [
            'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures',	'num_procedures',
            'num_medications', 'number_outpatient',	'number_emergency',	'number_inpatient',	'diag_1', 'diag_2',	'diag_3',
            'number_diagnoses',	'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
            'acarbose',	'miglitol',	'troglitazone',	'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
            'diabetesMed', 'readmitted'
        ]
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

def sample_balanced(dataset, ratio):
    X, y = dataset.X, dataset.y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=42)
    for train_idx, _ in sss.split(X, y):
        # 只保留按类别采样后的 X 和 y
        dataset.X = X[train_idx]
        dataset.y = y[train_idx]
        break

    return dataset


def get_diabetic_dataset(data_dir):
    train_set = DiabeticDataset(mode="train", data_dir=data_dir)
    val_set = DiabeticDataset(mode="val", data_dir=data_dir)
    test_set = DiabeticDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_diabetic_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = DiabeticDataset(mode="train", data_dir=data_dir)
    val_set = DiabeticDataset(mode="val", data_dir=data_dir)
    test_set = DiabeticDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_diabetic_dataloader(data_dir, batch_size):
    train_set = DiabeticDataset(mode="train", data_dir=data_dir)
    val_set = DiabeticDataset(mode="val", data_dir=data_dir)
    test_set = DiabeticDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_diabetic_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = DiabeticDataset(mode="train", data_dir=data_dir)
    val_set = DiabeticDataset(mode="val", data_dir=data_dir)
    test_set = DiabeticDataset(mode="test", data_dir=data_dir)
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
    data_dir = "/data/ruipeng/workdir/autoreg/.data/diabetic"
    sample_ratio = 0.2
    batch_size = 64

    # 原始数据
    train_set_full = DiabeticDataset(mode="train", data_dir=data_dir)
    # 采样后数据
    train_loader, val_loader, test_loader = get_diabetic_dataloader_sampled(
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