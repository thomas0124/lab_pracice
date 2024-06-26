# lab_pracice

## 1. DSC_Jaccard_Hausedorff
- Dice係数、Jaccard係数、Hausdorff距離を計算するプログラム

### 実行手順
- リポジトリのクローン
  
```bash
$git clone git@github.com:thomas0124/lab_pracice.git
```

- 仮想環境に入る
1. 現在のディレクトリを確認する

```bash
$pwd
```

2. 現在のディレクトリからlab_praciceの中のDSC_Jacard_Hausdorffに移動する

```bash
$cd ~/lab_pracice/DSC_Jacard_Hausdorff
```

3. 仮想環境に入る

```bash
$source .venv/bin/activate
```

4. 仮想環境に依存関係をインストールする

```bash
$pip install -r requirements.txt
```

5. プログラムを実行する

```bash
$python dice_index.py
$python jaccard_index.py
$python Hausdorff_index.py
```

6. 画像データの変更
- imaeg1には正解画像
- image2には予測画像

7. 仮想環境から出る

```bash
$deactivate
```


