# インストールした torchmetrics のバージョンを確認
import torchmetrics
from torchmetrics.functional import accuracy

# インストールした PyTorch Lightning のバージョンを確認
import pytorch_lightning as pl

# エラーが出た際は、一度ランタイムを再起動してください
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import joblib

# データの読み込み（相関の低い項目を削除）
path_test = '/Users/user/Desktop/FinalMission1/breast-cancer adjusted.csv'
df_adjust = pd.read_csv(path_test)

#外れ値除去 3σ法
import pandas as pd

def remove_outliers_all_columns(df, threshold=3):

    df_filtered = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            upper_limit = mean + threshold * std
            lower_limit = mean - threshold * std
            df_filtered = df_filtered[(df_filtered[col] >= lower_limit) & (df_filtered[col] <= upper_limit)]
    return df_filtered

# Remove outliers from all numeric columns
df_clean = remove_outliers_all_columns(df_adjust)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df_clean['diagnosis'])

df_clean['diagnosis'] = le.transform(df_clean['diagnosis'])

# 出力変数（diagnosis）
target = df_clean['diagnosis']

# 入力変数
features = df_clean.drop(columns=['id', 'diagnosis'], errors='ignore')

#標準化とデータ拡張
from sklearn.preprocessing import StandardScaler

def augment_data(features, target, n_samples=500):
    augmented_features = []
    augmented_target = []

    # 特徴量を標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # 標準化

    # 特徴量を標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # 標準化

    # スケーラーを保存
    joblib.dump(scaler, 'scaler.pkl')  # ここで保存


    for _ in range(n_samples):
        noise = np.random.normal(0, 0.05, features_scaled.shape)  # ノイズを適用
        new_features = features_scaled + noise
        augmented_features.append(new_features)
        augmented_target.extend(target)

    # 拡張データをテンソルに変換
    augmented_features = torch.tensor(np.vstack(augmented_features), dtype=torch.float32)
    augmented_target = torch.tensor(np.array(augmented_target), dtype=torch.int64)

    return augmented_features, augmented_target

x, t = augment_data(features, target, n_samples=500)


# 入力値と目標値をまとめる
dataset = torch.utils.data.TensorDataset(x, t)

# 各データセットのサンプルサイズを決定
# train : val: test = 70%　: 20% : 10%
n_train = int(len(dataset) * 0.7)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

# ランダムに分割を行うため、シードを固定して再現性を確保
pl.seed_everything(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

# バッチサイズの定義
batch_size = 30

# Data Loader を用意
# shuffle はデフォルトで False のため、訓練データのみ True に指定
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


#ネットワークの構築
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(18)  # Batch Normalization
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.5)  # ドロップアウト追加

    def forward(self, x):
        h = self.bn(x)  # 正規化
        h = self.fc1(h)  # 全結合層1
        h = F.relu(h)  # 活性化関数
        h = self.dropout(h)  # ドロップアウト適用
        h = self.fc2(h)  # 全結合層2
        return h


    # 学習データに対する処理
    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=True, on_epoch=True, prog_bar=True)
        return loss


    # 検証データに対する処理
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=False, on_epoch=True)
        return loss



    # テストデータに対する処理
    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=2, top_k=1), on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer





# 乱数のシードを固定
pl.seed_everything(0)

# 学習の実行
net = Net()
logger = CSVLogger(save_dir='logs', name='my_exp')
trainer = pl.Trainer(callbacks=[EarlyStopping(monitor='val_acc')], max_epochs=30, accelerator="cpu", deterministic=False, logger=logger)
trainer.fit(net, train_loader, val_loader)

# テストデータで検証
results = trainer.test(dataloaders=test_loader)


#logデータの保存
log = pd.read_csv('/Users/user/Desktop/FinalMission1/logs/my_exp/version_0/metrics.csv')

# 学習済みモデルを保存
joblib.dump(net, 'model.pkl')

# 学習後のモデルを保存
# torch.save(net.state_dict(), 'breast-cancer-updated.pt')