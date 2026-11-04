"""Lab 08 - From-scratch MLP trained on UCI Optical Digits (8x8, 10 classes)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "lines.linewidth": 2, "savefig.bbox": "tight",
})
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

LINES = []
def log(text=""):
    print(text)
    LINES.append(str(text))


# ------------------------------------------------------------------ building blocks
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)        # subtract the max for numerical safety
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, k=10):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1
    return Y


class MLP:
    """
    input (64) -> hidden layer (sigmoid) -> output layer (softmax over 10 digits)
    Loss: cross-entropy  L = -(1/n) sum_i sum_k  y_ik log p_ik
    """
    def __init__(self, n_in, n_hidden, n_out, lr=0.5, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(1 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        h = sigmoid(X @ self.W1 + self.b1)      # hidden activations
        p = softmax(h @ self.W2 + self.b2)      # class probabilities
        return h, p

    def backward(self, X, h, p, Y):
        """Backpropagation: apply the chain rule from the loss back to every weight."""
        n = len(X)
        delta2 = (p - Y) / n                    # dL/dz2 (softmax + cross-entropy)
        gW2, gb2 = h.T @ delta2, delta2.sum(0)
        delta1 = (delta2 @ self.W2.T) * h * (1 - h)   # back through W2, then through the sigmoid
        gW1, gb1 = X.T @ delta1, delta1.sum(0)
        # gradient-descent update
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1

    def fit(self, X, y, epochs=100, batch_size=32, X_val=None, y_val=None, seed=SEED):
        rng = np.random.default_rng(seed)
        Y = one_hot(y)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch_size):
                rows = order[s:s + batch_size]
                h, p = self.forward(X[rows])
                self.backward(X[rows], h, p, Y[rows])
            self._record(X, y, "train")
            if X_val is not None:
                self._record(X_val, y_val, "val")
        return self

    def _record(self, X, y, split):
        _, p = self.forward(X)
        self.history[f"{split}_loss"].append(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
        self.history[f"{split}_acc"].append(np.mean(p.argmax(1) == y))

    def predict(self, X):
        return self.forward(X)[1].argmax(1)


# ------------------------------------------------------------------ metrics
def confusion_matrix(y, yhat, k=10):
    M = np.zeros((k, k), int)
    np.add.at(M, (y, yhat), 1)
    return M


def per_class_scores(M):
    tp = np.diag(M).astype(float)
    prec, rec = tp / M.sum(0), tp / M.sum(1)
    return prec, rec, 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ experiment
def main():
    train = pd.read_csv(os.path.join(HERE, "..", "data", "optdigits_train.csv"))
    test = pd.read_csv(os.path.join(HERE, "..", "data", "optdigits_test.csv"))
    X_all = train.drop(columns="digit").to_numpy(float) / 16.0     # scale pixels to 0..1
    y_all = train["digit"].to_numpy()
    Xte = test.drop(columns="digit").to_numpy(float) / 16.0
    yte = test["digit"].to_numpy()

    idx = np.random.default_rng(SEED).permutation(len(y_all))
    n_val = int(0.15 * len(y_all))
    val, tr = idx[:n_val], idx[n_val:]
    Xtr, ytr, Xval, yval = X_all[tr], y_all[tr], X_all[val], y_all[val]
    log(f"Digits: {len(Xtr)} train / {len(Xval)} validation / {len(Xte)} test images, 64 inputs, 10 classes")

    # 1) choose the hidden-layer size on the validation set
    log("\nHidden-layer size study (100 epochs, lr = 0.5, batch 32):")
    val_acc = {}
    for H in [4, 8, 16, 32, 64, 128]:
        m = MLP(64, H, 10).fit(Xtr, ytr, epochs=100, X_val=Xval, y_val=yval)
        val_acc[H] = m.history["val_acc"][-1]
        log(f"   {H:4d} hidden units -> train acc {m.history['train_acc'][-1]:.4f}, "
            f"validation acc {val_acc[H]:.4f}, parameters {64 * H + H + H * 10 + 10}")
    best_H = max(val_acc, key=lambda h: (round(val_acc[h], 3), -h))
    log(f"Chosen hidden size: {best_H}")

    # 2) final network
    model = MLP(64, best_H, 10).fit(Xtr, ytr, epochs=100, X_val=Xval, y_val=yval)
    pred = model.predict(Xte)
    M = confusion_matrix(yte, pred)
    prec, rec, f1 = per_class_scores(M)
    log("\nTest-set results")
    log(f"  Accuracy        : {np.mean(pred == yte):.4f}  ({np.sum(pred != yte)} mistakes out of {len(yte)})")
    log(f"  Macro precision : {prec.mean():.4f}")
    log(f"  Macro recall    : {rec.mean():.4f}")
    log(f"  Macro F1        : {f1.mean():.4f}")
    log("  Per-digit F1    : " + "  ".join(f"{d}:{f:.3f}" for d, f in enumerate(f1)))
    off = M.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], i, j) for i in range(10) for j in range(10)), reverse=True)[:4]
    log("  Most common confusions: " + ", ".join(f"{i} read as {j} ({n}x)" for n, i, j in pairs))

    # --- figure 1: learning curves
    h = model.history
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    axes[0].plot(ep, h["train_loss"], color=BLUE, label="Training")
    axes[0].plot(ep, h["val_loss"], color=ORANGE, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)
    axes[1].plot(ep, h["train_acc"], color=BLUE, label="Training")
    axes[1].plot(ep, h["val_acc"], color=ORANGE, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.85, 1.005)
    fig.savefig(os.path.join(FIG, "learning_curves.pdf"))
    plt.close(fig)

    # --- figure 2: confusion matrix
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.imshow(M, cmap=SEQ)
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > M.max() / 2 else "#0b0b0b")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("Actual digit")
    fig.savefig(os.path.join(FIG, "confusion_matrix.pdf"))
    plt.close(fig)

    # --- figure 3: some of the mistakes
    wrong = np.where(pred != yte)[0][:12]
    fig, axes = plt.subplots(2, 6, figsize=(6.4, 2.6))
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(Xte[i].reshape(8, 8), cmap="Greys")
        ax.set_title(f"true {yte[i]}, said {pred[i]}", fontsize=7.5)
        ax.axis("off")
    fig.savefig(os.path.join(FIG, "mistakes.pdf"))
    plt.close(fig)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
