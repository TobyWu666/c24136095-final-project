# 資料集放置說明（問題三 QSVM）

本資料夾用於存放**問題三（QSVM 分類）**所使用的資料集。

## 一、放置 Moodle 提供的資料

若要使用 Moodle 提供的訓練/測試資料重新執行問題三，請於重跑前將檔案放入本資料夾：

```text
data/moodle_train.csv
data/moodle_test.csv
```

程式也接受下列備用檔名：

```text
data/train.csv
data/test.csv
data/training.csv
data/testing.csv
```

## 二、CSV 格式說明

- **含表頭（header）時：** 標籤欄位的名稱可為 `label`、`target`、`y`、`class` 或 `species`；若皆無，則自動以最後一欄作為標籤。
- **不含表頭時：** 以最後一欄作為標籤。

## 三、後備機制

若本資料夾內找不到任何上述檔案，`src/qsvm_iris.py` 會自動改用 `sklearn.load_iris()` 內建資料集，並以可重現的分層 70/30 切分（隨機種子固定）進行訓練與測試。實際使用的資料來源會記錄於 `outputs/data/qsvm_results.json` 之中，以利查核。
