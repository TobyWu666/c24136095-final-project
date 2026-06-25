"""
Final Project - Problems 1 & 2
  P1: noisy 2-qubit Grover for f(3)=1  -> state tomography -> rho' -> fidelity F(rho, rho')
  P2: noisy 3-qubit QFT on (|0>+|7>)/sqrt(2) -> state tomography -> density matrix -> negativity

Noisy fake backend : FakeManilaV2  (noise model via AerSimulator.from_backend)
Shots              : 8192 per measurement setting
Tomography         : manual Pauli-basis tomography (linear inversion + projection to nearest
                     physical state), Hilbert-space basis ordered |0..0>, ..., |1..1> (q_{n-1}..q_0)
"""
import warnings; warnings.filterwarnings("ignore")
import json, itertools
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
MPL_DIR = ROOT / ".matplotlib-cache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

FIG_DIR = ROOT / "outputs" / "figures"
DATA_DIR = ROOT / "outputs" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = 8192
SEED  = 42
np.random.seed(SEED)

backend = FakeManilaV2()
sim     = AerSimulator.from_backend(backend)     # carries the device noise model
print("Fake backend:", backend.name, "| n_qubits:", backend.num_qubits)

# Single-qubit Pauli matrices
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PMAT = {"I": I2, "X": PX, "Y": PY, "Z": PZ}

def kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

# ----------------------------------------------------------------------
# Manual Pauli-basis state tomography
# ----------------------------------------------------------------------
def add_basis_rotation(qc, qubit, basis):
    """Rotate so that measuring Z gives the <basis> eigenvalue."""
    if basis == "X":
        qc.h(qubit)
    elif basis == "Y":
        qc.sdg(qubit); qc.h(qubit)
    # Z -> nothing

def run_tomography(prep_circuit, n):
    """Return dict: measurement-setting (e.g. 'XZ') -> counts dict.
    Qubit ordering in setting string: position 0 = qubit 0 ... position n-1 = qubit n-1."""
    settings = ["".join(s) for s in itertools.product("XYZ", repeat=n)]
    circuits, names = [], []
    for setting in settings:
        qc = prep_circuit.copy()
        for q, b in enumerate(setting):
            add_basis_rotation(qc, q, b)
        qc.measure_all()
        circuits.append(transpile(qc, sim, optimization_level=1, seed_transpiler=SEED))
        names.append(setting)
    result = sim.run(circuits, shots=SHOTS, seed_simulator=SEED).result()
    return {name: result.get_counts(i) for i, name in enumerate(names)}

def expectation_from_counts(counts, active_qubits):
    """<Pauli> for the given measurement counts, multiplying eigenvalues only on active_qubits.
    Bitstring from qiskit is 'q_{n-1}...q_1 q_0' (little-endian)."""
    total = sum(counts.values()); val = 0.0
    for bitstr, c in counts.items():
        bits = bitstr.replace(" ", "")
        sign = 1
        for q in active_qubits:
            if bits[-1 - q] == "1":   # qubit q
                sign *= -1
        val += sign * c
    return val / total

def reconstruct_density_matrix(counts_by_setting, n):
    """Linear-inversion rho from all 4^n Pauli expectation values."""
    labels = ["".join(p) for p in itertools.product("IXYZ", repeat=n)]
    exps = {}
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)
    for label in labels:
        if label == "I" * n:
            exp = 1.0
        else:
            active = [q for q, p in enumerate(label) if p != "I"]   # q indexed 0..n-1
            # any measurement setting whose basis matches label on active qubits works
            comps = []
            for setting, counts in counts_by_setting.items():
                if all(setting[q] == label[q] for q in active):
                    comps.append(expectation_from_counts(counts, active))
            exp = float(np.mean(comps))
        exps[label] = exp
        # Pauli matrix with proper ordering: label[0]=qubit0 is the LEFT-most kron factor of
        # the operator on |q0 q1 ...> ; we want basis ordering |q_{n-1}..q_0>, so reverse.
        mats = [PMAT[p] for p in label][::-1]
        rho += exp * kron_list(mats)
    rho /= dim
    return rho, exps

def project_to_physical(rho):
    """Nearest physical density matrix (Smolin 2012): Hermitian, PSD, trace 1."""
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals = vals[::-1]; vecs = vecs[:, ::-1]
    d = len(vals); lam = vals.copy(); a = 0.0
    i = d - 1
    while i >= 0:
        if lam[i] + a / (i + 1) < 0:
            a += lam[i]; lam[i] = 0.0; i -= 1
        else:
            break
    for j in range(i + 1):
        lam[j] = lam[j] + a / (i + 1)
    rho_phys = (vecs * lam) @ vecs.conj().T
    return (rho_phys + rho_phys.conj().T) / 2

def partial_transpose_np(rho, n, qubitsA):
    """Partial transpose over subsystem A (list of qubit indices 0..n-1).
    Basis ordering |q_{n-1}..q_0>: row axis for qubit q is (n-1-q)."""
    t = rho.reshape([2] * n + [2] * n)
    perm = list(range(2 * n))
    for q in qubitsA:
        ai = n - 1 - q
        bi = n + (n - 1 - q)
        perm[ai], perm[bi] = perm[bi], perm[ai]
    return np.transpose(t, perm).reshape(2 ** n, 2 ** n)

def negativity(rho, n, subsystem_dims, mask):
    """Negativity = sum of |negative eigenvalues| of the partial transpose over A=mask."""
    pt = partial_transpose_np(rho, n, mask)
    ev = np.linalg.eigvalsh((pt + pt.conj().T) / 2)
    return float(np.sum(np.abs(ev[ev < 0])))

def plot_density(rho, title, path):
    dim = rho.shape[0]; n = int(np.log2(dim))
    labels = [format(i, f"0{n}b") for i in range(dim)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    for ax, data, lab in zip(axes, [rho.real, rho.imag], ["Re", "Im"]):
        im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{lab}(rho)"); ax.set_xticks(range(dim)); ax.set_yticks(range(dim))
        ax.set_xticklabels(labels, rotation=90, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
        for i in range(dim):
            for j in range(dim):
                v = data[i, j]
                if abs(v) > 0.08:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def serializable_counts(counts_by_setting):
    return {
        setting: {bits: int(count) for bits, count in sorted(counts.items())}
        for setting, counts in sorted(counts_by_setting.items())
    }

def rounded_expectations(exps):
    return {label: round(float(value), 6) for label, value in sorted(exps.items())}

results = {}

# ======================================================================
# PROBLEM 1 : 2-qubit Grover, f(3)=1  -> marked state |11> = |3>
# ======================================================================
print("\n" + "=" * 60 + "\nPROBLEM 1: 2-qubit Grover (f(3)=1)\n" + "=" * 60)
g = QuantumCircuit(2, name="Grover f(3)=1")
g.h([0, 1])                       # uniform superposition
g.barrier(label="oracle")
g.cz(0, 1)                        # oracle: phase-flip |11>
g.barrier(label="diffuser")
g.h([0, 1]); g.x([0, 1])          # diffuser
g.cz(0, 1)
g.x([0, 1]); g.h([0, 1])

# ideal output
psi_ideal = Statevector(g)
rho_ideal = DensityMatrix(psi_ideal).data
print("Ideal output state (should be |11>=|3>):")
print(np.round(psi_ideal.data, 3))

g.draw("mpl", fold=-1)
plt.savefig(FIG_DIR / "fig_p1_circuit.png", dpi=150, bbox_inches="tight"); plt.close()

counts1 = run_tomography(g, 2)
rho1_li, exps1 = reconstruct_density_matrix(counts1, 2)
rho1 = project_to_physical(rho1_li)

# desired rho = |3><3|
rho_target = np.zeros((4, 4), dtype=complex); rho_target[3, 3] = 1.0
F1 = state_fidelity(DensityMatrix(rho_target), DensityMatrix(rho1))   # (Tr sqrt(sqrt(rho) rho' sqrt(rho)))^2
neg1 = negativity(rho1, 2, [2, 2], [0])

print(f"\nFidelity  F(rho, rho') = {F1:.4f}")
print(f"Negativity (A={{q0}}|B={{q1}}) = {neg1:.4f}")
print("rho'(1,1) population of |11>:", round(rho1[3, 3].real, 4))
plot_density(rho1, "Problem 1: reconstructed density matrix (noisy Grover)", FIG_DIR / "fig_p1_rho.png")

# transpiled depth / layout info
tg = transpile(g.copy(), sim, optimization_level=1, seed_transpiler=SEED)
results["p1"] = dict(
    ideal_state="|11> = |3>",
    fidelity=round(F1, 4), negativity=round(neg1, 4),
    pop_11=round(float(rho1[3, 3].real), 4),
    transpiled_depth=tg.depth(), n_settings=9, shots=SHOTS,
    pauli_expectations=rounded_expectations(exps1),
    counts_ZZ=counts1["ZZ"],
)

# ======================================================================
# PROBLEM 2 : 3-qubit QFT on (|0>+|7>)/sqrt(2)
# ======================================================================
print("\n" + "=" * 60 + "\nPROBLEM 2: 3-qubit QFT on (|0>+|7>)/sqrt(2)\n" + "=" * 60)

from qiskit.circuit.library import QFT   # canonical QFT (verified == analytic DFT_8)

q = QuantumCircuit(3, name="QFT on (|0>+|7>)/sqrt2")
# prepare (|000>+|111>)/sqrt2
q.h(0); q.cx(0, 1); q.cx(1, 2)
q.barrier(label="QFT")
q.compose(QFT(3), [0, 1, 2], inplace=True)

psi2_ideal = Statevector(q)
rho2_ideal = DensityMatrix(psi2_ideal).data
print("Ideal QFT output amplitudes:")
print(np.round(psi2_ideal.data, 3))

q.decompose().draw("mpl", fold=-1)
plt.savefig(FIG_DIR / "fig_p2_circuit.png", dpi=150, bbox_inches="tight"); plt.close()

counts2 = run_tomography(q, 3)
rho2_li, exps2 = reconstruct_density_matrix(counts2, 3)
rho2 = project_to_physical(rho2_li)

F2  = state_fidelity(DensityMatrix(rho2_ideal), DensityMatrix(rho2))
neg2_a = negativity(rho2, 3, [2, 2, 2], [0])        # A={q0} | B={q1,q2}
neg2_b = negativity(rho2, 3, [2, 2, 2], [1])        # A={q1} | B={q0,q2}
neg2_c = negativity(rho2, 3, [2, 2, 2], [2])        # A={q2} | B={q0,q1}
print(f"\nFidelity to ideal QFT output = {F2:.4f}")
print(f"Negativity  A={{q0}}|B={{q1,q2}} = {neg2_a:.4f}")
print(f"Negativity  A={{q1}}|B={{q0,q2}} = {neg2_b:.4f}")
print(f"Negativity  A={{q2}}|B={{q0,q1}} = {neg2_c:.4f}")
plot_density(rho2, "Problem 2: reconstructed density matrix (noisy QFT)", FIG_DIR / "fig_p2_rho.png")

tq = transpile(q.copy(), sim, optimization_level=1, seed_transpiler=SEED)
results["p2"] = dict(
    ideal_state="QFT[(|000>+|111>)/sqrt2]",
    fidelity=round(F2, 4),
    negativity_q0=round(neg2_a, 4), negativity_q1=round(neg2_b, 4), negativity_q2=round(neg2_c, 4),
    transpiled_depth=tq.depth(), n_settings=27, shots=SHOTS,
    pauli_expectations=rounded_expectations(exps2),
    counts_ZZZ=counts2["ZZZ"],
)

# numeric dump of density matrices
np.set_printoptions(precision=3, suppress=True)
with open(DATA_DIR / "density_matrices.txt", "w") as f:
    f.write("Basis ordering: |q2 q1 q0> = |000>,|001>,...,|111>\n\n")
    f.write("PROBLEM 1 - reconstructed rho' (noisy Grover), basis |q1 q0>:\n")
    f.write(np.array2string(np.round(rho1, 3)) + "\n")
    f.write(f"\nFidelity F(rho=|3><3|, rho') = {F1:.4f}\nNegativity = {neg1:.4f}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("PROBLEM 2 - reconstructed rho' (noisy QFT), basis |q2 q1 q0>:\n")
    f.write(np.array2string(np.round(rho2, 3)) + "\n")
    f.write(f"\nFidelity to ideal QFT output = {F2:.4f}\n")
    f.write(f"Negativity A={{q0}}|B={{q1,q2}} = {neg2_a:.4f}\n")
    f.write(f"Negativity A={{q1}}|B={{q0,q2}} = {neg2_b:.4f}\n")
    f.write(f"Negativity A={{q2}}|B={{q0,q1}} = {neg2_c:.4f}\n")

with open(DATA_DIR / "results_p12.json", "w") as f:
    json.dump(results, f, indent=2)

tomography_data = {
    "metadata": {
        "backend": backend.name,
        "shots_per_setting": SHOTS,
        "seed": SEED,
        "measurement_basis_order": "setting string position 0 = qubit 0",
        "bitstring_order": "qiskit counts use q_{n-1}...q_0",
        "density_matrix_basis_order": "|q_{n-1}...q_0> = |0...0>, |0...1>, ..., |1...1>",
    },
    "p1": {
        "experiment": "noisy 2-qubit Grover for f(3)=1",
        "tomography_settings": 9,
        "counts_by_setting": serializable_counts(counts1),
        "pauli_expectations": rounded_expectations(exps1),
    },
    "p2": {
        "experiment": "noisy 3-qubit QFT on (|0>+|7>)/sqrt(2)",
        "tomography_settings": 27,
        "counts_by_setting": serializable_counts(counts2),
        "pauli_expectations": rounded_expectations(exps2),
    },
}
with open(DATA_DIR / "tomography_p12.json", "w") as f:
    json.dump(tomography_data, f, indent=2)

print("\nSaved figures to", FIG_DIR)
print("Saved density_matrices.txt + results_p12.json + tomography_p12.json to", DATA_DIR)
print("SUMMARY:", json.dumps({"P1_fidelity": results["p1"]["fidelity"],
                              "P1_negativity": results["p1"]["negativity"],
                              "P2_fidelity": results["p2"]["fidelity"],
                              "P2_neg_q0": results["p2"]["negativity_q0"]}))
