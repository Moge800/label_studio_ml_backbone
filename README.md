# Label Studio ML Backend - YOLOv11

Label StudioでYOLOv11を使った物体検出を行うためのMLバックエンドです。

## 概要

このプロジェクトは、Label Studioのアノテーション作業を支援するために、YOLOv11モデルを使用して自動的に物体検出を行い、バウンディングボックスの予測を提供します。

## 主な機能

- YOLOv11モデルによる物体検出
- Label Studioとの認証付き画像取得
- バウンディングボックス座標の自動正規化
- 信頼度スコア付きの予測結果
- 一時ファイルの自動クリーンアップ

## 必要な環境

- Python 3.8以上
- Label Studio
- YOLOv11学習済みモデル（.ptファイル）

## セットアップ

1. **依存関係のインストール**
   ```bash
   pip install -r requirements.txt
   ```

2. **設定ファイルの作成**
   
   `hidden_key.py`ファイルを作成し、以下の設定を記述してください：
   ```python
   # Label Studioのホスト（ポート番号も含む）
   LABEL_STUDIO_HOST = "http://localhost:8080"
   
   # Label StudioのAPIキー
   LABEL_STUDIO_API_KEY = "your_api_key_here"
   
   # YOLOモデルファイルのパス
   MODEL_PATH = "path/to/your/model.pt"
   
   # クラスIDとラベル名のマッピング
   CLASS_NAMES = {
       0: "person",
       1: "car",
       2: "bicycle"
       # 必要に応じて追加
   }
   ```

3. **MLバックエンドの初期化**（Windows）
   ```powershell
   .\setup.ps1
   ```

## 使用方法

1. **MLバックエンドの起動**（Windows）
   ```powershell
   .\launch.ps1
   ```

2. **Label Studioでの設定**
   - Label Studioの管理画面でMLバックエンドを追加
   - エンドポイント: `http://localhost:9090`
   - プロジェクトにMLバックエンドを接続

3. **アノテーション作業**
   - 画像をアップロードすると自動的にYOLOv11による予測が実行されます
   - 予測結果を確認・修正してアノテーションを完成させてください

## ファイル構成

- `yolo_backend.py` - メインのMLバックエンド実装
- `requirements.txt` - 必要なPythonパッケージ
- `setup.ps1` - MLバックエンド初期化スクリプト
- `launch.ps1` - MLバックエンド起動スクリプト
- `hidden_key.py` - 設定ファイル（要作成）

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルをご覧ください。