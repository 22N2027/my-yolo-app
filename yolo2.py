import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# ページ設定（ブラウザのタブ名など）
st.set_page_config(page_title="YOLO物体検出アプリ", layout="wide")

# タイトル
st.title("物体検出アプリケーション")

# --- サイドバー設定 ---
st.sidebar.header("モデル設定")

# モデルの選択肢を作成
# 'concrete.pt' がフォルダにない場合は選択肢から外す処理を入れています
model_options = ["yolov8n.pt"]  # デフォルトモデル
if os.path.exists("concrete.pt"):
    model_options.append("concrete.pt")  # 自作モデルがあれば追加
elif os.path.exists("last.pt"):
    model_options.append("last.pt")

selected_model = st.sidebar.selectbox("使用するモデルを選択", model_options)

# 信頼度のしきい値（スライダーで調整できるようにすると便利です）
conf_threshold = st.sidebar.slider("信頼度しきい値 (Confidence)", 0.0, 1.0, 0.25, 0.05)

# --- モデルのロード関数 ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 選択されたモデルをロード
try:
    model = load_model(selected_model)
    st.sidebar.success(f"モデルロード完了: {selected_model}")
except Exception as e:
    st.sidebar.error(f"モデルの読み込みに失敗しました: {e}")

# --- メイン画面 ---
st.subheader(f"現在のモデル: {selected_model}")

uploaded_file = st.file_uploader("画像をアップロードしてください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像をPIL形式で読み込み
    image = Image.open(uploaded_file)

    # 物体検出の実行
    # conf=conf_threshold でスライダーの値を適用します
    results = model.predict(source=image, conf=conf_threshold, stream=False)

    # 検出結果の描画
    annotated_image = results[0].plot()

    # 画像を横並びに表示
    col1, col2 = st.columns(2)

    with col1:
        st.header("オリジナル画像")
        st.image(image, use_column_width=True)

    with col2:
        st.header("検出後の画像")
        st.image(annotated_image, channels="BGR", use_column_width=True)
        
        # （オプション）検出された物体の数やクラス名を表示する
        boxes = results[0].boxes
        if boxes:
            st.info(f"検出数: {len(boxes)} 個")
            # クラスIDを名前に変換してリスト表示（例: ['cat', 'dog']）
            class_names = [model.names[int(cls)] for cls in boxes.cls]
            st.write("検出されたクラス:", set(class_names)) # setで重複削除
        else:
            st.warning("物体は検出されませんでした。")