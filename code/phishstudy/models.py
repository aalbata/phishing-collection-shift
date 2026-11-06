"""Classifiers compared in the study, all CPU-only and seeded.

Numeric models read the engineered features after log1p and standardisation
fitted on the fitting split. The text model reads the normalised message text
through a TF-IDF vocabulary fitted on the fitting split. Neural models stop
early on tuning-split average precision. Every model exposes ``fit`` and
``score`` (probability-like score, larger means more likely phishing).
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn

from .features import FEATURES, FORMAT_FEATURES, URGENCY_FEATURES, URL_FEATURES

# variant name -> (model family, feature columns or None for text)
VARIANTS = {
    "logistic": ("logistic", FEATURES),
    "random_forest": ("random_forest", FEATURES),
    "gradient_boosting": ("gradient_boosting", FEATURES),
    "mlp": ("mlp", FEATURES),
    "single_token_attention": ("single_token_attention", FEATURES),
    "feature_token_transformer": ("feature_token_transformer", FEATURES),
    "ftt_no_url": ("feature_token_transformer", [f for f in FEATURES if f not in URL_FEATURES]),
    "ftt_no_urgency": ("feature_token_transformer", [f for f in FEATURES if f not in URGENCY_FEATURES]),
    "gb_format_only": ("gradient_boosting", FORMAT_FEATURES),
    "tfidf_logistic": ("tfidf_logistic", None),
}
# Two declared settings per family (index 0 and 1); weighting is a separate factor.
SETTINGS = {
    "logistic": [dict(C=0.1), dict(C=1.0)],
    "random_forest": [dict(min_samples_leaf=1), dict(min_samples_leaf=5)],
    "gradient_boosting": [dict(max_leaf_nodes=15), dict(max_leaf_nodes=31)],
    "mlp": [dict(width=32), dict(width=64)],
    "single_token_attention": [dict(width=32), dict(width=64)],
    "feature_token_transformer": [dict(width=32), dict(width=64)],
    "tfidf_logistic": [dict(C=1.0), dict(C=10.0)],
}


class MLP(nn.Module):
    def __init__(self, p: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p, width), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(width, width), nn.ReLU(), nn.Dropout(0.1), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SingleTokenAttention(nn.Module):
    """All features projected into ONE token before multi-head self-attention.

    With a single key the softmax weight is exactly 1, so the attention block
    reduces to its value and output projections: an explicitly labelled control.
    """

    def __init__(self, p: int, width: int, heads: int = 4):
        super().__init__()
        self.project = nn.Linear(p, width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        z = self.project(x).unsqueeze(1)
        a, _ = self.attn(z, z, z, need_weights=False)
        return self.head(a[:, 0]).squeeze(-1)


class FeatureTokenTransformer(nn.Module):
    """One token per feature (t_j = x_j w_j + b_j) plus a [CLS] token, pre-norm encoder blocks.

    Follows the numerical-feature tokenisation of Gorishniy et al. (2021); depth,
    width and feed-forward form are reduced for CPU use, so it is not their exact
    FT-Transformer configuration. No attention mask is needed: every feature is
    observed for every message.
    """

    def __init__(self, p: int, width: int, heads: int = 4, blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(p, width))
        self.bias = nn.Parameter(torch.empty(p, width))
        bound = 1 / np.sqrt(width)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        nn.init.uniform_(self.cls, -bound, bound)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=2 * width,
                                           dropout=dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))

    def tokens(self, x):
        z = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), z], dim=1)

    def forward(self, x):
        return self.head(self.encoder(self.tokens(x))[:, 0]).squeeze(-1)


class TextFeatures:
    """TF-IDF vocabulary fitted once on the fitting split; matrices cached per named frame."""

    def __init__(self, fit_frame):
        start = time.perf_counter()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                          sublinear_tf=True, dtype=np.float32)
        self.vectorizer.fit(fit_frame.text)
        # The set of pruned n-grams is kept by scikit-learn only for introspection and can hold millions of
        # entries; it is not used by transform.
        self.vectorizer.stop_words_ = None
        self.fit_seconds = time.perf_counter() - start
        self.cache = {}

    def matrix(self, frame):
        key = frame.attrs.get("name")
        if key is None:
            return self.vectorizer.transform(frame.text)
        if key not in self.cache:
            self.cache[key] = self.vectorizer.transform(frame.text)
        return self.cache[key]


NETS = {"mlp": MLP, "single_token_attention": SingleTokenAttention,
        "feature_token_transformer": FeatureTokenTransformer}


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class Classifier:
    def __init__(self, variant: str, setting: int, weighting: str, seed: int, threads: int = 2,
                 max_epochs: int = 60, patience: int = 8, batch_size: int = 256, lr: float = 1e-3,
                 text=None):
        self.variant, self.setting, self.weighting, self.seed = variant, setting, weighting, seed
        self.family, self.columns = VARIANTS[variant]
        self.params = SETTINGS[self.family][setting]
        self.threads, self.max_epochs, self.patience = threads, max_epochs, patience
        self.batch_size, self.lr = batch_size, lr
        self.history, self.n_parameters = [], None
        self.text = text  # shared TextFeatures for the text model

    # numeric preprocessing: log1p then standardisation fitted on the fitting split only
    def _numeric(self, frame, fit=False):
        X = np.log1p(frame[self.columns].to_numpy(dtype=np.float64))
        if fit:
            self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X).astype(np.float32)

    def fit(self, fit_frame, tune_frame):
        set_seed(self.seed, self.threads)
        y = fit_frame.label.to_numpy(dtype=int)
        yv = tune_frame.label.to_numpy(dtype=int)
        w = compute_sample_weight("balanced", y) if self.weighting == "balanced" else np.ones(len(y))
        start = time.perf_counter()
        if self.family == "tfidf_logistic":
            X = self.text.matrix(fit_frame)
            self.model = LogisticRegression(C=self.params["C"], solver="liblinear", max_iter=5000,
                                            random_state=self.seed)
            self.model.fit(X, y, sample_weight=w)
            self.n_parameters = int(X.shape[1] + 1)
        else:
            X = self._numeric(fit_frame, fit=True)
            if self.family == "logistic":
                self.model = LogisticRegression(C=self.params["C"], max_iter=5000, random_state=self.seed)
                self.model.fit(X, y, sample_weight=w)
                self.n_parameters = int(X.shape[1] + 1)
            elif self.family == "random_forest":
                self.model = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                    min_samples_leaf=self.params["min_samples_leaf"],
                                                    n_jobs=self.threads, random_state=self.seed)
                self.model.fit(X, y, sample_weight=w)
                self.n_parameters = int(sum(t.tree_.node_count for t in self.model.estimators_))
            elif self.family == "gradient_boosting":
                self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                            max_leaf_nodes=self.params["max_leaf_nodes"],
                                                            l2_regularization=1.0, early_stopping=False,
                                                            random_state=self.seed)
                self.model.fit(X, y, sample_weight=w)
                self.n_parameters = int(sum(tree.get_n_leaf_nodes() * 2 - 1 for it in self.model._predictors
                                            for tree in it))
            else:
                self._fit_net(X, y, w, self._numeric(tune_frame), yv)
        self.fit_seconds = time.perf_counter() - start
        self.tune_ap = float(average_precision_score(yv, self.score(tune_frame)))
        return self

    def _fit_net(self, X, y, w, V, yv):
        self.model = NETS[self.family](X.shape[1], self.params["width"])
        self.n_parameters = int(sum(p.numel() for p in self.model.parameters()))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        data = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.tensor(y, dtype=torch.float32),
                                              torch.tensor(w / w.mean(), dtype=torch.float32))
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                             generator=torch.Generator().manual_seed(self.seed))
        best, state, stale = -np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total = 0.0
            for xb, yb, wb in loader:
                opt.zero_grad(set_to_none=True)
                loss = (nn.functional.binary_cross_entropy_with_logits(self.model(xb), yb, reduction="none")
                        * wb).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                total += float(loss.detach()) * len(xb)
            ap = float(average_precision_score(yv, self._net_scores(V)))
            self.history.append(dict(epoch=epoch, fit_loss=total / len(y), tune_ap=ap))
            if ap > best + 1e-6:
                best, state, stale = ap, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model.load_state_dict(state)
        self.best_epoch = int(max(self.history, key=lambda h: (h["tune_ap"], -h["epoch"]))["epoch"])

    def _net_scores(self, X):
        self.model.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(X), 2048):
                out.append(torch.sigmoid(self.model(torch.from_numpy(X[i:i + 2048]))).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.empty(0)

    def score(self, frame) -> np.ndarray:
        if self.family == "tfidf_logistic":
            return self.model.predict_proba(self.text.matrix(frame))[:, 1]
        return self.score_matrix(self._numeric(frame))

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        """Scores of an already transformed numeric matrix (numeric families only)."""
        if isinstance(self.model, nn.Module):
            return self._net_scores(X)
        return self.model.predict_proba(X)[:, 1]
