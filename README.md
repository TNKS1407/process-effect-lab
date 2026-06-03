# Process Effect Lab

「何を変えると何が起きるか」を確認するための Python ベースの学習・実験ツールです。静的な説明ではなく、パラメータを動かして数値・グラフ・診断結果がどう変わるかを見ます。

## 使い方

```bash
pip install -r requirements.txt
```

Notebook でじっくり進める場合：

```bash
jupyter lab whatif_modeling_lab.ipynb
```

スライダーで触る場合：

```bash
streamlit run app.py
```

代表的な感度分析を一括で回す場合：

```bash
python run_all.py --out outputs
```

## 入っているもの

- `whatif_modeling_lab.ipynb`：説明つきの実験 Notebook
- `app.py`：スライダーで動かす Streamlit アプリ
- `effect_lab_core.py`：計算本体。OLS、Ridge、PLS1、局所重み付き PLS、転移用特徴空間、診断などを NumPy/Pandas ベースで実装
- `run_all.py`：代表パラメータを動かした感度分析の一括実行
- `study_guide.md`：勉強用の説明、観察ポイント、練習問題
- `outputs/`：一括実行で生成される CSV と PNG

## 基本の見方

1. まず 1 つだけパラメータを変える。
2. 予測精度だけでなく、係数、条件数、VIF、有効サンプル数、診断フラグを見る。
3. 「良くなった / 悪くなった」で終わらせず、なぜそうなったかを言葉にする。
4. 最後に、現実の設備・運転・測定のどの状況に対応するかを考える。

## 重要な注意

このツールは学習・検証用です。実プラントや実験データにそのまま適用して意思決定するものではありません。実データでは、データの意味、測定限界、制御の有無、運転モード、品種、設備変更、欠損の理由を必ず確認してください。
