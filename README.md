# Phylotekton Ek TmAlign

**TmAlign — Local Duplex Stability Search** (https://github.com/c2997108/TmAlign) の解析結果を整理し、csv形式で出力するツールセットです。
さらに、作成されたcsvファイルから、プライマーの有効性を図示する系統樹を作成します。
テストデータはreleaseで提供しています。
https://github.com/Sacabambaeus/Tm_Align_analysis/releases/download/test_data/tm_result_mf-u_all_carangi.tsv
Tm_Align用のミトコンドリアデータベースもreleaseで提供しています。
https://github.com/Sacabambaeus/Tm_Align_analysis/releases/tag/database

## 📌 目的 (Purpose)
1. **集計 (`analyze_tm.py`):** TmAlignのTSV出力を読み込み、NCBI Taxonomyデータベースを用いて生物種情報を付与・集計します。
2. **可視化 (`tree_map.py`):** 集計データ（CSV）をもとに、生物種の包含関係とスコア（Tm値、Identity）を図示した系統樹マップを作成します。

## 📦 必要要件 (Requirements)

* **Python 3.9+** (Python 3.9.18で開発)
* **必須ライブラリ:**
  * `pandas` (データ処理用)
  * `matplotlib` (描画用)
  * 以下のコマンドでインストール可能です:
    ```bash
    pip install pandas matplotlib
    ```
* **NCBI Taxonomy Data** ...
* 
* **NCBI Taxonomy Data**
  以下のFTPサイトから2つのファイルをダウンロードし、**指定のディレクトリ構成**で配置してください。

| ファイル | URL | 配置方法 |
| :--- | :--- | :--- |
| **new_taxdump.tar.gz** | [Link](https://ftp.ncbi.nih.gov/pub/taxonomy/new_taxdump/) | 解凍して `nodes.dmp`, `names.dmp` を<br>`taxdump/` フォルダに入れてください。 |
| **nucl_gb.accession2taxid.gz** | [Link](https://ftp.ncbi.nih.gov/pub/taxonomy/accession2taxid/) | **解凍せずに** そのまま<br>`acc2taxid/` フォルダに入れてください。 |

### 推奨ディレクトリ構成
```text
.
├── analyze_tm.py
├── tree_map.py
├── taxdump/
│   ├── names.dmp
│   └── nodes.dmp
└── acc2taxid/
    └── nucl_gb.accession2taxid.gz
```

## 🚀 実行手順 (Usage)

### 1. 解析と集計 (analyze_tm.py)
入力TSVファイルを読み込み、Taxonomy情報を付与して階級ごとの集計ファイルを出力します。

#### 最小実行例
（※概念的なコマンドです。実際には出力オプションが必須となるため、下の「実行例」を推奨します）
```bash
python3 analyze_tm.py test_input.tsv --acc2taxid acc2taxid --taxdump taxdump
```
### 実行例
```
python3 analyze_tm.py /path/to/tm_result.tsv \
    --acc2taxid /path/to/acc2taxid \
    --taxdump /path/to/taxdump \
    --c /path/to/class_summary.csv \
    --o /path/to/order_summary.csv \
    --f /path/to/family_summary.csv \
    --a /path/to/accession_taxonomy.csv
```

### 📝 オプション詳細 (Options)

#### 1. analyze_tm.py のオプション

| オプション | 必須 | 説明 |
| :--- | :--- | :--- |
| `--acc2taxid` | **YES** | `nucl_gb.accession2taxid.gz` が入っているディレクトリを指定します。 |
| `--taxdump` | **YES** | `nodes.dmp`, `names.dmp` が入っているディレクトリを指定します。 |
| `--c` | **YES** | **綱 (Class)** レベルの集計結果を出力するCSVパスを指定します。<br>例: `summary_class.csv` |
| `--o` | **YES** | **目 (Order)** レベルの集計結果を出力するCSVパスを指定します。<br>例: `summary_order.csv` |
| `--f` | **YES** | **科 (Family)** レベルの集計結果を出力するCSVパスを指定します。<br>例: `summary_family.csv` |
| `--a` | No | アクセッションIDごとにTaxonomy情報を付与した詳細リストを出力します。<br>指定しない場合は出力されません。 |

---


### 2. 可視化 (tree_map.py)
Step 1 で作成したCSV（例: class_summary.csv）を入力とし、系統樹マップを描画します。
#### 最小実行例
（※概念的なコマンドです。実際には出力オプションが必須となるため、下の「実行例」を推奨します）
```bash
python3 tree_map.py taxonomy_class_summary.csv output.png --taxdump taxdump
```
### 実行例
```
python3 tree_map.py /path/to/input.csv /path/to/output.pdf \
    --taxdump /path/to/taxdump \
    --a \
    --t Actinopteri \
    --d family \
    tree_act_fam.pdf
```

#### 2. tree_map.py のオプション

**基本引数**
`python3 tree_map.py [入力CSV] [出力画像] [オプション]`

| 引数・オプション | 必須 | 説明 |
| :--- | :--- | :--- |
| `[入力CSV]` | **YES** | `analyze_tm.py` で出力された集計ファイル（例: `summary_class.csv`）。 |
| `[出力画像]` | **YES** | 出力ファイル名。拡張子で `.png` または `.pdf` を指定します。<br>※巨大な系統樹になる場合は **.pdf** を推奨します。<br>※現在 `.png` では、系統樹が背景からはみ出してしまいます。系統樹のサイズにかかわらず、**.pdf**を推奨します。 |
| `--taxdump` | **YES** | `nodes.dmp` 等が入っているディレクトリを指定します。 |

**フィルター設定（表示する生物種を絞り込む）**
| オプション | 説明 |
| :--- | :--- |
| `--a` | **動物界 (Animalia)** のみを表示します。 |
| `--p` | **植物界 (Plantae)** のみを表示します。 |
| `--f` | **菌界 (Fungi)** のみを表示します。 |
| `--m` | **原核生物 (Monera/Bacteria/Archaea)** のみを表示します。 |
| `--r` | **その他 (Rest)**。動物・植物・菌以外の真核生物（原生生物など）を表示します。 |
| `--taxon <名前>` | 特定の分類群（例: `Actinopteri`）以下のみを表示します。カンマ区切りで複数指定も可能です。 |

**表示設定**
| オプション | 説明 |
| :--- | :--- |
| `--t <名前>` | 系統樹の **根 (Root)** にする分類群を指定します。 |

| `--d <階級>` | 描画する **末端 (Leaf)** の階級を指定します。<br>`class` (綱), `order` (目), `family` (科) から選択可能です。 |

**出力される図(系統樹)**
 * プライマーはMiFish-U、以下の条件で実行
 ```
 python3 tree_map.py mf-u_all_carangi_family.csv --taxdump taxdump --a --t Carangiformes --d family mf-u_all_carangi_carangi_fam.pdf
```
 * 赤色の図形がTm値、水色の図形がidentity(単純な塩基の一致率)、円がForwardプライマー、ひし形がReverseプライマー、右端の緑色の四角形と数字は、計算に使用したレコードの数。一つのTaxIDにつき、一つのレコードとして使用。
<img width="1209" height="599" alt="carangiformes_family" src="https://github.com/user-attachments/assets/6088f0bf-efd0-49e3-934c-aab99e916174" />
























