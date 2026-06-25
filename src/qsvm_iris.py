"""
Final Project - Problem 3
Two ideal, noiseless kernel-based QSVMs on the Iris dataset (3-class).
  Model A: ZZ-feature map  (Qiskit built-in ZZFeatureMap)
  Model B: Hardware-efficient ansatz (HEA) feature map
Kernel: FidelityStatevectorKernel (exact statevector -> ideal / noiseless)
Classifier: sklearn SVC(kernel="precomputed")
Metric: test-set accuracy
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import numpy as np
from pathlib import Path

RNG = 42
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
MPL_DIR = ROOT / ".matplotlib-cache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DATA_DIR = ROOT / "outputs" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(RNG)

# ----------------------------------------------------------------------
# 1) Load dataset. Prefer Moodle-provided train/test CSV files.
# ----------------------------------------------------------------------
TRAIN_CANDIDATES = [
    DATA_DIR / "moodle_train.csv",
    DATA_DIR / "train.csv",
    DATA_DIR / "training.csv",
]
TEST_CANDIDATES = [
    DATA_DIR / "moodle_test.csv",
    DATA_DIR / "test.csv",
    DATA_DIR / "testing.csv",
]

def first_existing(paths):
    return next((path for path in paths if path.exists()), None)

def encode_labels(y_train_raw, y_test_raw=None):
    combined = y_train_raw if y_test_raw is None else np.concatenate([y_train_raw, y_test_raw])
    classes, encoded = np.unique(combined.astype(str), return_inverse=True)
    y_train = encoded[:len(y_train_raw)]
    if y_test_raw is None:
        return y_train, list(classes)
    y_test = encoded[len(y_train_raw):]
    return y_train, y_test, list(classes)

def load_csv_dataset(path):
    with path.open("r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
    first_tokens = [token.strip() for token in first_line.split(",")]
    has_header = False
    for token in first_tokens:
        try:
            float(token)
        except ValueError:
            has_header = True
            break

    if has_header:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8-sig")
        if arr.ndim == 0:
            arr = arr.reshape(1)
        columns = list(arr.dtype.names)
        target_names = {"label", "target", "y", "class", "species"}
        target_col = next((col for col in columns if col.lower() in target_names), columns[-1])
        feature_cols = [col for col in columns if col != target_col]
        X = np.column_stack([arr[col].astype(float) for col in feature_cols])
        y_raw = np.asarray(arr[target_col]).astype(str)
        return X, y_raw, feature_cols

    arr = np.genfromtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    X = arr[:, :-1]
    y_raw = arr[:, -1].astype(str)
    feature_cols = [f"x{i}" for i in range(X.shape[1])]
    return X, y_raw, feature_cols

train_path = first_existing(TRAIN_CANDIDATES)
test_path = first_existing(TEST_CANDIDATES)

if train_path and test_path:
    X_train_raw, y_train_raw, feat_names = load_csv_dataset(train_path)
    X_test_raw, y_test_raw, _ = load_csv_dataset(test_path)
    y_train, y_test, class_names = encode_labels(y_train_raw, y_test_raw)
    dataset_source = "Moodle CSV train/test files"
    dataset_detail = {
        "train_path": str(train_path.relative_to(ROOT)),
        "test_path": str(test_path.relative_to(ROOT)),
    }
else:
    data = load_iris()
    X_raw, y_raw = data.data, data.target          # X:(150,4)  y in {0,1,2}
    feat_names = data.feature_names
    class_names = list(data.target_names)          # setosa / versicolor / virginica
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.30, stratify=y_raw, random_state=RNG
    )
    dataset_source = "sklearn.load_iris fallback; Moodle train/test CSV files were not found"
    dataset_detail = {
        "expected_train_files": [str(path.relative_to(ROOT)) for path in TRAIN_CANDIDATES],
        "expected_test_files": [str(path.relative_to(ROOT)) for path in TEST_CANDIDATES],
    }

n_qubits = X_train_raw.shape[1]

# Scale the 4 features into [0, pi] so they act as rotation angles.
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("dataset source:", dataset_source)
print("n_qubits =", n_qubits)
print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("train class counts:", np.unique(y_train, return_counts=True)[1].tolist())
print("test  class counts:", np.unique(y_test,  return_counts=True)[1].tolist())

# ----------------------------------------------------------------------
# 2) Feature maps
# ----------------------------------------------------------------------
# Model A: ZZ-feature map (built-in)
zz_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="linear")

# Model B: hardware-efficient ansatz feature map (data-encoding)
def build_hea_feature_map(n_qubits, reps=2, entanglement="linear"):
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for r in range(reps):
        for i in range(n_qubits):          # data-encoding layer
            qc.ry(x[i], i)
            qc.rz(x[i], i)
        if entanglement == "linear":        # entangling layer
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "circular":
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qc.cx(i, j)
        if r < reps - 1:
            qc.barrier()
    return qc

hea_map = build_hea_feature_map(n_qubits, reps=2, entanglement="linear")

# ----------------------------------------------------------------------
# 3) Quantum kernels (FidelityStatevectorKernel = ideal/noiseless) + SVM
# ----------------------------------------------------------------------
def run_qsvm(feature_map, name):
    qk = FidelityStatevectorKernel(feature_map=feature_map)
    K_train = qk.evaluate(X_train)              # (Ntr, Ntr)
    K_test  = qk.evaluate(X_test, X_train)      # (Nte, Ntr)
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    print(f"\n===== {name} =====")
    print("Test accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4, output_dict=True)
    return dict(name=name, acc=acc, cm=cm, K_train=K_train, y_pred=y_pred, report=report)

res_zz  = run_qsvm(zz_map,  "QSVM - ZZ-feature map")
res_hea = run_qsvm(hea_map, "QSVM - HEA feature map")

# ----------------------------------------------------------------------
# 4) Figures for the report
# ----------------------------------------------------------------------
# Circuit diagrams
zz_map.decompose().draw("mpl", fold=24)
plt.savefig(FIG_DIR / "fig_circuit_zz.png", dpi=150, bbox_inches="tight"); plt.close()
hea_map.draw("mpl", fold=24)
plt.savefig(FIG_DIR / "fig_circuit_hea.png", dpi=150, bbox_inches="tight"); plt.close()

# Confusion matrices
def plot_cm(cm, title, path):
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right"); plt.yticks(ticks, class_names)
    th = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > th else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

plot_cm(res_zz["cm"],  "ZZ-feature map QSVM",  FIG_DIR / "fig_cm_zz.png")
plot_cm(res_hea["cm"], "HEA feature map QSVM", FIG_DIR / "fig_cm_hea.png")

# Kernel matrices (train)
for res, tag in [(res_zz, "zz"), (res_hea, "hea")]:
    plt.figure(figsize=(4.6, 3.8))
    plt.imshow(res["K_train"], aspect="auto", cmap="viridis")
    plt.colorbar(); plt.title(f"Quantum Kernel Matrix (Train) - {tag.upper()}")
    plt.xlabel("train index"); plt.ylabel("train index")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"fig_kernel_{tag}.png", dpi=150); plt.close()

# Accuracy comparison bar
plt.figure(figsize=(4.6, 3.8))
names = ["ZZ-feature map", "HEA"]
accs  = [res_zz["acc"], res_hea["acc"]]
bars = plt.bar(names, accs, color=["#4C72B0", "#C44E52"])
plt.ylim(0, 1.05); plt.ylabel("Test accuracy"); plt.title("QSVM Test Accuracy Comparison")
for b, a in zip(bars, accs):
    plt.text(b.get_x()+b.get_width()/2, a+0.01, f"{a:.4f}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_acc_compare.png", dpi=150); plt.close()

qsvm_results = {
    "dataset_source": dataset_source,
    "dataset_detail": dataset_detail,
    "feature_names": list(feat_names),
    "class_names": class_names,
    "n_qubits": int(n_qubits),
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "feature_scaling": "MinMaxScaler(feature_range=(0, pi)); fit on train, transform test",
    "models": {
        "zz_feature_map": {
            "accuracy": round(float(res_zz["acc"]), 6),
            "confusion_matrix": res_zz["cm"].astype(int).tolist(),
            "classification_report": res_zz["report"],
        },
        "hea_feature_map": {
            "accuracy": round(float(res_hea["acc"]), 6),
            "confusion_matrix": res_hea["cm"].astype(int).tolist(),
            "classification_report": res_hea["report"],
        },
    },
}
with open(OUT_DATA_DIR / "qsvm_results.json", "w") as f:
    json.dump(qsvm_results, f, indent=2)

print("\nSaved figures to", FIG_DIR)
print("Saved qsvm_results.json to", OUT_DATA_DIR)
print("FINAL  ZZ acc =", round(res_zz["acc"],4), " HEA acc =", round(res_hea["acc"],4))
