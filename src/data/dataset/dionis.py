import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader

# Dionis 元数据信息
class DionisMetaData:
    def __init__(self):
        super().__init__()
        self.dataset_name = "dionis"
        self.instance = 416188
        self.feats = 61
        self.in_features = 60
        self.out_features = 355
        self.batch_size = 128
        self.is_balanced = True
        self.class_ratio = None
        self.data_dir = "/.data/dionis"

        self.column_names = [f'V{i}' for i in range(1, 61)] + ['class']
        self.label_name = 'class'
        self.continuous_features = [f'V{i}' for i in range(1, 61)]

        self.means = {
            'V1': 8308.025067998116, 'V2': -117.5739209203533, 'V3': 10015.29548905783, 'V4': -5081.156926677367,
            'V5': 80.41936817015387, 'V6': 10070.770135131239, 'V7': 933.5927249223909, 'V8': -5624.798490105433,
            'V9': 130.50143204513344, 'V10': 38.57154218766519, 'V11': 6055.942525973839, 'V12': 4010.9801556027564,
            'V13': -5852.296695243495, 'V14': 0.0, 'V15': -6956.660434707392, 'V16': 8180.395198804386,
            'V17': 7033.108741722491, 'V18': -3182.4827577921515, 'V19': 78.86629359808548, 'V20': 167.61009928205522,
            'V21': 71.63139254375426, 'V22': 8882.739543187214, 'V23': -3742.9603328303556, 'V24': -1526.0456019875633,
            'V25': 29.728245408325083, 'V26': 84.1218824185224, 'V27': 0.0, 'V28': 2238.025930589061,
            'V29': 7132.543999346449, 'V30': 2328.538119792017, 'V31': 126.72827424144857, 'V32': 8663.656818553154,
            'V33': 0.0, 'V34': 341.7865796226705, 'V35': 0.0, 'V36': 1691.9750040846925,
            'V37': 0.0, 'V38': 6396.879746652955, 'V39': 108.32669851124972, 'V40': 1844.452088959797,
            'V41': -1375.322498486261, 'V42': -1227.6816366642, 'V43': 818.8409252549328, 'V44': -715.5324420694494,
            'V45': 54.65668399857756, 'V46': -7668.606867569464, 'V47': 9245.83445942699, 'V48': 7005.3025820062085,
            'V49': 42.79700039405269, 'V50': -585.0167976971945, 'V51': 7531.959657654714, 'V52': 400.55544369371535,
            'V53': 9.309819120205292, 'V54': 0.0, 'V55': 7787.801308543255, 'V56': -8372.240597998982,
            'V57': -4846.513433832787, 'V58': 135.62115438215423, 'V59': 718.9164560246811, 'V60': 33.85317933241708
        }

        self.stds = {
            'V1': 1121.463167057501, 'V2': 1351.451856823899, 'V3': 3193.5475092818356, 'V4': 3226650.755954017,
            'V5': 466.2976904407399, 'V6': 986.7182821996723, 'V7': 406280.0455331893, 'V8': 2554388.281963558,
            'V9': 577.4684753426584, 'V10': 206.6121495858779, 'V11': 1506.5689512198173, 'V12': 1533974.4310525937,
            'V13': 3608.930596516671, 'V14': 0.0, 'V15': 2548917.7943341085, 'V16': 1619.1795423512683,
            'V17': 1941.406664893007, 'V18': 2575732.683960475, 'V19': 1333.7950438708406, 'V20': 394.82637102737306,
            'V21': 328.3789470381447, 'V22': 1526.5686786677577, 'V23': 3602781.2686927975, 'V24': 2624239.647936929,
            'V25': 158.40908138054738, 'V26': 276.8832187187364, 'V27': 0.0, 'V28': 2169.1284363337795,
            'V29': 1861.2721102951905, 'V30': 583723.7297933527, 'V31': 928.2444634566858, 'V32': 1370.9247562054945,
            'V33': 0.0, 'V34': 1397.3434328658009, 'V35': 0.0, 'V36': 1121.4631646680282,
            'V37': 0.0, 'V38': 3436.0212123951674, 'V39': 893.1346179773345, 'V40': 807156.4830319155,
            'V41': 3813969.598047053, 'V42': 3340530.663450589, 'V43': 771593.6785601602, 'V44': 553.0158332325823,
            'V45': 238.4569865628242, 'V46': 4007456.7542995955, 'V47': 1227.941317918852, 'V48': 3500316.370766258,
            'V49': 192.24375386713703, 'V50': 1265950.7900515313, 'V51': 1680.1385964727829, 'V52': 643.71799770103,
            'V53': 1.6899205948796683, 'V54': 0.0, 'V55': 1793.7853647599254, 'V56': 3934768.506495391,
            'V57': 2026182.9777809582, 'V58': 645.4590407244762, 'V59': 689499.3489356012, 'V60': 169.16953226954263
        }


class DionisProcessor:
    def __init__(self):
        super().__init__()
        self.meta = DionisMetaData()

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

class DionisDataset(Dataset):
    def __init__(self, data_dir, mode: str = "train"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.processor = DionisProcessor()

        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        self.column_names = [f'V{i}' for i in range(1, 61)] + ['class']

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

def get_dionis_dataset(data_dir):
    train_set = DionisDataset(mode="train", data_dir=data_dir)
    val_set = DionisDataset(mode="val", data_dir=data_dir)
    test_set = DionisDataset(mode="test", data_dir=data_dir)
    return train_set, val_set, test_set

def get_dionis_dataset_sampled(data_dir, sample_ratio=0.2):
    train_set = DionisDataset(mode="train", data_dir=data_dir)
    val_set = DionisDataset(mode="val", data_dir=data_dir)
    test_set = DionisDataset(mode="test", data_dir=data_dir)
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    return train_set, val_set, test_set

def get_dionis_dataloader(data_dir, batch_size):
    train_set = DionisDataset(mode="train", data_dir=data_dir)
    val_set = DionisDataset(mode="val", data_dir=data_dir)
    test_set = DionisDataset(mode="test", data_dir=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_dionis_dataloader_sampled(data_dir, batch_size, sample_ratio=0.2):
    train_set = DionisDataset(mode="train", data_dir=data_dir)
    val_set = DionisDataset(mode="val", data_dir=data_dir)
    test_set = DionisDataset(mode="test", data_dir=data_dir)
    # 分层按比例采样（推荐）
    train_set = sample_balanced(train_set, sample_ratio)
    val_set = sample_balanced(val_set, sample_ratio)
    test_set = sample_balanced(test_set, sample_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("---- DIONIS Sampling Test ----")

    data_dir = "/data/ruipeng/workdir/autoreg/.data/dionis"  # ⚠️改成你的路径
    sample_ratio = 0.2
    batch_size = 64

    # ① 原始数据
    train_set_full = DionisDataset(mode="train", data_dir=data_dir)

    # ② 采样后数据
    train_loader, val_loader, test_loader = get_dionis_dataloader_sampled(
        data_dir=data_dir,
        batch_size=batch_size,
        sample_ratio=sample_ratio
    )
    train_set_sampled = train_loader.dataset

    # ====== 打印采样前类别分布 ======
    y_full = train_set_full.y.numpy()

    print("\n====== 采样前类别数量 ======")
    import collections

    full_counter = collections.Counter(y_full)
    for cls, num in full_counter.items():
        print(f"class {cls}: {num}")

    # ====== 打印采样后类别分布 ======
    y_sampled = train_set_sampled.y.numpy()

    print("\n====== 采样后类别数量 ======")
    sampled_counter = collections.Counter(y_sampled)
    for cls, num in sampled_counter.items():
        print(f"class {cls}: {num}")

    # ====== 理论采样数量 ======
    print("\n====== 理论采样数量（按比例） ======")
    for cls, num in full_counter.items():
        print(f"class {cls} 应为: {int(num * sample_ratio)}")

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
