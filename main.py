import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pytorch_lightning as pl

# モデルの定義
class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(18)
        self.fc1 = nn.Linear(18, 10)
        self.fc2 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h = self.bn(x)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h

# モデルとスケーラーのロード
@st.cache_resource  # Streamlitキャッシュを利用
def load_model_and_scaler():
    model = joblib.load('model.pkl')  # 学習済みモデルをロード
    scaler = joblib.load('scaler.pkl')  # 学習済みスケーラーをロード
    model.eval()  # モデルを評価モードに設定
    return model, scaler

# 入力スライダーを作成する関数
def create_input_sliders():
    st.sidebar.header("入力パラメータ")
    inputs = {
        "radius_mean": st.sidebar.slider('半径の平均値 (radius_mean) (mm)', 0.0, 29.0, step=0.1),
        "perimeter_mean": st.sidebar.slider('周囲長の平均値 (perimeter_mean) (mm)', 0.0, 189.0, step=0.1),
        "area_mean": st.sidebar.slider('面積の平均値 (area_mean) (mm²)', 0.0, 2501.0, step=1.0),
        "compactness_mean": st.sidebar.slider('コンパクト度の平均値 (compactness_mean)', 0.0, 1.0, step=0.01),
        "concavity_mean": st.sidebar.slider('凹度の平均値 (concavity_mean)', 0.0, 1.0, step=0.01),
        "concave_points_mean": st.sidebar.slider('凹点の平均値 (concave points_mean)', 0.0, 1.0, step=0.01),
        "radius_se": st.sidebar.slider('半径の標準誤差 (radius_se) (mm)', 0.0, 3.0, step=0.1),
        "perimeter_se": st.sidebar.slider('周囲長の標準誤差 (perimeter_se) (mm)', 0.0, 22.0, step=0.1),
        "area_se": st.sidebar.slider('面積の標準誤差 (area_se) (mm²)', 0.0, 543.0, step=1.0),
        "compactness_se": st.sidebar.slider('コンパクト度の標準誤差 (compactness_se)', 0.0, 1.0, step=0.01),
        "concavity_se": st.sidebar.slider('凹度の標準誤差 (concavity_se)', 0.0, 1.0, step=0.01),
        "concave_points_se": st.sidebar.slider('凹点の標準誤差 (concave points_se)', 0.0, 1.0, step=0.01),
        "radius_worst": st.sidebar.slider('最悪値の半径 (radius_worst) (mm)', 0.0, 37.0, step=0.1),
        "perimeter_worst": st.sidebar.slider('最悪値の周囲長 (perimeter_worst) (mm)', 0.0, 252.0, step=1.0),
        "area_worst": st.sidebar.slider('最悪値の面積 (area_worst) (mm²)', 0.0, 4254.0, step=1.0),
        "compactness_worst": st.sidebar.slider('最悪値のコンパクト度 (compactness_worst)', 0.0, 2.0, step=0.01),
        "concavity_worst": st.sidebar.slider('最悪値の凹度 (concavity_worst)', 0.0, 2.0, step=0.01),
        "concave_points_worst": st.sidebar.slider('最悪値の凹点 (concave points_worst)', 0.0, 1.0, step=0.01),
    }
    return np.array([[v for v in inputs.values()]])

# Streamlit アプリ
st.title("乳がん診断アプリ")
st.write("スライダーでデータを入力し、乳がんが良性か悪性かを診断します。")

# モデルとスケーラーをロード
model, scaler = load_model_and_scaler()

# 入力フォーム
input_data = create_input_sliders()

# 診断ボタン
if st.button('診断を実行'):
    # 入力データを標準化
    input_data_scaled = scaler.transform(input_data)
    input_data_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    # モデルで予測
    output = model(input_data_tensor)
    prediction = torch.argmax(output, dim=1).item()

    # 結果を診断に変換
    diagnosis = "良性" if prediction == 0 else "悪性"

    # 結果を表示
    st.success(f"予測結果: {diagnosis}")

