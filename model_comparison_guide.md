# 外観検査AI ベンチマーク - 検証モデル比較ガイド

## 目次
1. [全体マップ](#1-全体マップ)
2. [既存モデル（自作）](#2-既存モデル自作)
3. [追加候補モデル（Anomalib）](#3-追加候補モデルanomalib)
4. [性能比較表](#4-性能比較表)
5. [用途別おすすめ](#5-用途別おすすめ)
6. [利用可能データセット](#6-利用可能データセット)

---

## 1. 全体マップ

```
                          精度 (MVTec AD AUROC)
                    100% ┤
    PatchCore ●──────────┤ 99.6%   ← メモリバンク方式
   EfficientAD ●─────────┤ 99.1%   ← 先生-生徒方式（産業向き本命）
  Reverse Dist ●─────────┤ 98.5%   ← 逆蒸留方式
         DRAEM ●──────────┤ 98.0%   ← 合成異常+再構成
        PaDiM  ●──────────┤ 97.5%   ← ガウス分布方式
      FastFlow ●──────────┤ 96.3%   ← 正規化フロー方式
                     95% ┤
                          │
       Deep AE ●──────────┤ ~90%    ← 4層Conv+SSIM（本ベンチマーク）
      STFPM    ●──────────┤ 89%     ← 軽量先生-生徒方式
     Sparse AE ●──────────┤ ~85%    ← 2層Conv+L1（本ベンチマーク）
                     80% ┤
                          └─────────────────────────────→ 推論速度
                          遅い                            速い
                          PatchCore  FastFlow  PaDiM  STFPM  EfficientAD
                          (~100ms)   (~50ms)  (~20ms) (~5ms)  (<2ms)
```

---

## 2. 既存モデル（自作）

### 2.1 Sparse Autoencoder (SAE) - スパースオートエンコーダ

**アーキテクチャ：**
```
入力(1,128,128)
  → Conv2d(1→32, k=4, s=2) + BatchNorm + ReLU
  → Conv2d(32→64, k=4, s=2) + BatchNorm + ReLU
  → FC(65536 → 64)  ← ボトルネック（潜在次元64）
  → FC(64 → 65536)
  → ConvTranspose2d(64→32) + BatchNorm + ReLU
  → ConvTranspose2d(32→1) + Sigmoid
出力(1,128,128)
```

**損失関数：** `MSE(入力, 再構成) + λ × L1(潜在表現)`  (λ=0.001)

**コンセプト：**
- 画像を圧縮→復元し、復元誤差で異常を検出
- L1正則化で潜在表現をスパース（まばら）にする → 少数の重要な特徴だけで表現
- パラメータ数: ~22K-50K（非常に軽量）

**強み：** エッジデバイス向き、解釈しやすい
**弱み：** 表現力が低い、微細な異常を見逃しやすい

---

### 2.2 Deep Autoencoder (DAE) - ディープオートエンコーダ

**アーキテクチャ：**
```
入力(1,128,128)
  → Enc1: Conv×2(1→32) + BatchNorm + LeakyReLU(0.2)  → (32,64,64)
  → Enc2: Conv×2(32→64) + BatchNorm + LeakyReLU(0.2) → (64,32,32)
  → Enc3: Conv×2(64→128) + BatchNorm + LeakyReLU(0.2) → (128,16,16)
  → Enc4: Conv×2(128→256) + BatchNorm + LeakyReLU(0.2) → (256,8,8)
  → FC(16384 → 128)  ← ボトルネック（潜在次元128）
  → FC(128 → 16384)
  → Dec4〜Dec1: ConvTranspose + Conv + BatchNorm + ReLU
出力(1,128,128)
```

**損失関数：** `MSE(入力, 再構成) + 0.1 × (1 - SSIM(入力, 再構成))`

**コンセプト：**
- SAEより深い4層構造でより複雑な特徴を学習
- SSIM損失で構造的類似性も考慮 → 人間の知覚に近い品質評価
- パラメータ数: ~200K-500K（SAEの約10倍）

**強み：** 微細な異常の検出力向上、SSIM損失で構造情報を保持
**弱み：** 学習に時間がかかる、SAEよりメモリ消費大

---

### SAE vs DAE の本質的な違い

| 観点 | SAE | DAE |
|------|-----|-----|
| 設計思想 | 少ない特徴で表現（スパース性） | 深い階層で表現（深層性） |
| ボトルネック | 64次元 | 128次元 |
| エンコーダ層数 | 2層 | 4層（各層2重Conv） |
| 正則化 | L1（潜在変数のスパース化） | SSIM（知覚的品質） |
| パラメータ比 | 1x | ~10x |
| 想定用途 | エッジ/組込み | サーバー/高精度 |

---

## 3. 追加候補モデル（Anomalib）

> 以下のモデルはすべて **Anomalib v2.2.0**（インストール済み）で利用可能です。
> ImageNet等で事前学習済みのバックボーンを活用するため、自作AEより大幅に高精度です。

---

### 3.1 PatchCore（パッチコア）

**カテゴリ：** メモリバンク方式
**論文：** CVPR 2022 "Towards Total Recall in Industrial Anomaly Detection"
**MVTec AD AUROC：** 99.6%

**仕組み：**
```
【学習フェーズ】
正常画像群
  → ImageNet学習済みバックボーン（WideResNet-50等）で特徴抽出
  → 中間層の特徴マップからパッチ特徴を収集
  → Coreset Subsampling（代表的なパッチだけ選抜）
  → メモリバンクに保存

【検査フェーズ】
テスト画像
  → 同じバックボーンで特徴抽出
  → 各パッチ特徴とメモリバンクのk近傍距離を計算
  → 距離が大きいパッチ = 異常領域
```

**なぜ高精度？**
- ImageNetで学習済みの強力な特徴表現を利用（自分で特徴を学ぶ必要がない）
- パッチレベルで比較するため、微小な異常も捉えられる
- Coreset Subsamplingでメモリ効率を改善

**コード例：**
```python
from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine

model = Patchcore(
    backbone="wide_resnet50_2",  # バックボーン選択
    coreset_sampling_ratio=0.1,  # メモリの10%に圧縮
)
```

**初中級者へ：** 「正常品の写真帳を作って、テスト品がどれだけ写真帳と違うか」で判定するモデル。写真帳（メモリバンク）の品質が精度を決めます。

---

### 3.2 EfficientAD

**カテゴリ：** 先生-生徒（Student-Teacher）方式 + オートエンコーダ
**論文：** WACV 2024 "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies"
**MVTec AD AUROC：** 99.1%

**仕組み：**
```
【学習フェーズ】
先生モデル（Teacher）：ImageNet学習済み、固定（学習しない）
生徒モデル（Student）：先生の出力を模倣するように学習
オートエンコーダ（AE）：先生の特徴を再構成するように学習

【検査フェーズ】
テスト画像 → 先生の出力 F_t
          → 生徒の出力 F_s
          → AEの出力   F_ae

異常スコア = |F_t - F_s| + |F_t - F_ae|

正常品: 先生≈生徒≈AE → スコア小
異常品: 先生≠生徒, 先生≠AE → スコア大
```

**なぜ速い？**
- PDN（Patch Description Network）という超軽量バックボーンを使用
- 先生と生徒がともに小さなネットワーク
- GPU不要でCPUでもリアルタイム推論可能

**コード例：**
```python
from anomalib.models import EfficientAd

model = EfficientAd(
    teacher_out_channels=384,
    model_size="small",  # "small" or "medium"
)
```

**初中級者へ：** 「熟練検査員（先生）と新人（生徒）」モデル。新人は正常品なら先生と同じ判断ができるが、異常品では先生と意見が食い違う。その食い違いの大きさで異常を検出します。

---

### 3.3 PaDiM（パディム）

**カテゴリ：** 埋め込み + ガウス分布方式
**論文：** ICPR 2021 "PaDiM: a Patch Distribution Modeling Framework"
**MVTec AD AUROC：** 97.5%（WideResNet-50使用時）

**仕組み：**
```
【学習フェーズ】
正常画像群
  → ImageNet学習済みバックボーンで特徴抽出
  → 画像の各位置(i,j)ごとに特徴ベクトルを収集
  → 各位置で多変量ガウス分布 N(μ_ij, Σ_ij) を推定

【検査フェーズ】
テスト画像
  → 同じバックボーンで特徴抽出
  → 各位置の特徴ベクトルとガウス分布のマハラノビス距離を計算
  → 距離が大きい位置 = 異常領域

マハラノビス距離 = √((x-μ)ᵀ Σ⁻¹ (x-μ))
```

**なぜ効率的？**
- 学習は統計量の計算のみ（勾配降下なし）→ 非常に速い
- メモリバンクではなく分布パラメータを保存 → メモリ効率が良い
- ランダム次元選択で計算量を削減

**コード例：**
```python
from anomalib.models import Padim

model = Padim(
    backbone="wide_resnet50_2",
    n_features=550,  # ランダム選択する特徴次元数
)
```

**初中級者へ：** 「正常品の各部位の"ばらつき範囲"を統計的に覚えて、範囲外なら異常」というモデル。品質管理の管理図（X-R管理図）の高次元版です。

---

### 3.4 STFPM（Student-Teacher Feature Pyramid Matching）

**カテゴリ：** 先生-生徒方式
**論文：** arXiv 2021 "Student-Teacher Feature Pyramid Matching"
**MVTec AD AUROC：** 89%（anomalib実装）

**仕組み：**
```
【学習フェーズ】
先生（ResNet学習済み、固定）
生徒（同じ構造、ランダム初期化）

正常画像 → 先生の特徴ピラミッド {F_t1, F_t2, F_t3}
        → 生徒の特徴ピラミッド {F_s1, F_s2, F_s3}
損失 = Σ ||F_ti - F_si||²  （生徒が先生を模倣）

【検査フェーズ】
テスト画像 → 先生と生徒の各レベル特徴の差 → 異常マップ
```

**コード例：**
```python
from anomalib.models import Stfpm

model = Stfpm(backbone="resnet18")
```

**初中級者へ：** EfficientADの先輩にあたるモデル。よりシンプルな先生-生徒方式で、「粗い特徴」から「細かい特徴」まで複数スケールで比較します。精度は劣りますが最も理解しやすい蒸留モデルです。

---

### 3.5 FastFlow（ファストフロー）

**カテゴリ：** 正規化フロー（Normalizing Flow）方式
**論文：** arXiv 2021 "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows"
**MVTec AD AUROC：** 96.3%（WideResNet-50使用時）

**仕組み：**
```
【学習フェーズ】
正常画像
  → バックボーンで特徴抽出
  → 正規化フロー（可逆ネットワーク）で特徴を正規分布N(0,I)に変換
  → 正常特徴がうまく正規分布に変換されるように学習

【検査フェーズ】
テスト画像
  → バックボーンで特徴抽出
  → 正規化フローで変換を試みる
  → 変換後の確率密度（log-likelihood）を計算
  → 確率が低い = 正規分布にうまく変換できない = 異常
```

**コード例：**
```python
from anomalib.models import Fastflow

model = Fastflow(
    backbone="wide_resnet50_2",
    flow_steps=8,  # フローの層数
)
```

**初中級者へ：** 「正常品の特徴は一つの"型"に変換できるはず。変換できないものは異常」というアプローチ。数学的に確率密度を推定するため、異常の不確実性まで定量化できるユニークなモデルです。

---

### 3.6 Reverse Distillation（逆蒸留）

**カテゴリ：** 蒸留 + デコーダ方式
**論文：** CVPR 2022 "Anomaly Detection via Reverse Distillation from One-Class Embedding"
**MVTec AD AUROC：** 98.5%

**仕組み：**
```
【従来の蒸留】  先生エンコーダ → 生徒エンコーダ（先生の出力を模倣）
【逆蒸留】      先生エンコーダ → ボトルネック → 生徒デコーダ（先生の特徴を復元）

先生（エンコーダ、固定）
  → 特徴マップ F_enc
  → ボトルネック（情報圧縮）
  → 生徒デコーダで F_enc を復元 → F_dec

異常スコア = |F_enc - F_dec|（復元できない部分 = 異常）
```

**コード例：**
```python
from anomalib.models import ReverseDistillation

model = ReverseDistillation(backbone="wide_resnet50_2")
```

**初中級者へ：** オートエンコーダと蒸留のハイブリッド。「先生が見た特徴を、生徒が復元できるか？」で異常を判定。SAE/DAEと同じ再構成アプローチですが、ImageNet特徴を使う点で大幅に強化されています。

---

## 4. 性能比較表

### 4.1 定量比較

| モデル | 方式 | MVTec AUROC | 推論速度 | メモリ使用量 | 学習の簡易さ |
|--------|------|-------------|----------|-------------|-------------|
| **PatchCore** | メモリバンク | **99.6%** | ~100ms | 大（メモリバンク） | ★★★★★（ほぼ学習不要） |
| **EfficientAD** | 先生-生徒+AE | 99.1% | **<2ms** | 小 | ★★★☆☆ |
| **Reverse Dist** | 逆蒸留 | 98.5% | ~15ms | 中 | ★★★☆☆ |
| **PaDiM** | ガウス分布 | 97.5% | ~20ms | 小 | ★★★★★（統計計算のみ） |
| **FastFlow** | 正規化フロー | 96.3% | ~50ms | 中 | ★★☆☆☆（要チューニング） |
| Deep AE (自作) | 再構成 | ~90% | ~5ms | 小 | ★★★★☆ |
| STFPM | 先生-生徒 | 89% | ~5ms | 小 | ★★★★☆ |
| Sparse AE (自作) | 再構成+L1 | ~85% | **<1ms** | 極小 | ★★★★★ |

### 4.2 手法カテゴリ別の特徴

| カテゴリ | モデル例 | 原理 | メリット | デメリット |
|----------|---------|------|---------|-----------|
| **再構成ベース** | SAE, DAE | 正常品を復元、復元誤差で検出 | 直感的、軽量 | 異常品も復元してしまう問題 |
| **メモリバンク** | PatchCore, PaDiM | 正常特徴を記憶、距離で検出 | 高精度、学習簡単 | メモリ消費、推論やや遅い |
| **先生-生徒** | EfficientAD, STFPM | 正常時の一致/異常時の不一致 | 高速、バランス良い | 学習が必要 |
| **正規化フロー** | FastFlow | 確率密度で検出 | 確率的解釈可能 | チューニング難度高 |
| **蒸留+再構成** | Reverse Dist | 特徴空間での再構成 | 高精度 | やや複雑 |

---

## 5. 用途別おすすめ

### シナリオA: 学習用ベンチマーク（本ノートブック推奨構成）

**推奨: SAE + DAE + PatchCore + EfficientAD + PaDiM（計5モデル）**

理由：
- SAE/DAE: 自作モデルのベースライン（既存）
- PatchCore: 精度の上限を知る（メモリバンク代表）
- EfficientAD: 産業用途の本命（先生-生徒代表）
- PaDiM: 統計的アプローチの代表（ガウス分布代表）
- 3つの異なるアプローチを比較でき、教育的価値が高い

### シナリオB: 包括的な手法比較

**推奨: 全8モデル**

追加で STFPM, FastFlow, Reverse Distillation も含め、全カテゴリをカバー。

### シナリオC: 産業導入検討

**推奨: EfficientAD + PatchCore**

- EfficientAD: リアルタイム検査ライン向け
- PatchCore: オフライン品質監査向け

---

## 6. 利用可能データセット

### VisA（ローカル済み、すぐ使える）

```
Anomalib/
  ├── pcb1/        (PCB基板1 - 1,004正常 + 100異常)
  ├── pcb2/        (PCB基板2)
  ├── pcb3/        (PCB基板3)
  ├── pcb4/        (PCB基板4)
  ├── capsules/    (カプセル)
  ├── candle/      (キャンドル)
  ├── cashew/      (カシューナッツ)
  ├── chewinggum/  (ガム)
  ├── fryum/       (揚げ菓子)
  ├── macaroni1/   (マカロニ1)
  ├── macaroni2/   (マカロニ2)
  └── pipe_fryum/  (パイプ揚げ菓子)
```

### MVTec AD（要ダウンロード、4.9GB）

15カテゴリ: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

### 合成データ（既存ノートブック）

128x128グレースケール、4種の異常（scratch, stain, missing, discolor）

---

## 参考リンク

- [Anomalib GitHub](https://github.com/open-edge-platform/anomalib)
- [Anomalib ドキュメント](https://anomalib.readthedocs.io/)
- [MVTec AD データセット](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [PatchCore 論文 (CVPR 2022)](https://arxiv.org/abs/2106.08265)
- [EfficientAD 論文 (WACV 2024)](https://arxiv.org/abs/2303.14535)
- [PaDiM 論文 (ICPR 2021)](https://arxiv.org/abs/2011.08785)
