"""Figures, each drawn only from files under evidence/ and datasets/.

Vector PDF plus 600 dpi PNG; serif type of 7 to 8 pt; one categorical order
used identically in every figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Fixed categorical order (validated palette); a model keeps its colour in every figure.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"feature_token_transformer": SLOTS[0], "mlp": SLOTS[1], "gradient_boosting": SLOTS[2],
               "tfidf_logistic": SLOTS[3], "single_token_attention": SLOTS[4], "random_forest": SLOTS[5],
               "logistic": SLOTS[6], "gb_format_only": SLOTS[7], "ftt_no_url": "#5598e7",
               "ftt_no_urgency": "#184f95"}
MODEL_LABEL = {"logistic": "Logistic regression", "random_forest": "Random forest",
               "gradient_boosting": "Gradient boosting", "mlp": "MLP",
               "single_token_attention": "Single-token attention", "feature_token_transformer": "Feature-token Transformer",
               "ftt_no_url": "FTT without link features", "ftt_no_urgency": "FTT without urgency count",
               "gb_format_only": "Gradient boosting, format only", "tfidf_logistic": "TF-IDF logistic"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
FEATURE_LABEL = {"characters": "characters", "words": "words", "unique_words": "distinct words", "line_count": "lines",
                 "mean_word_length": "mean word length", "upper_fraction": "upper-case fraction",
                 "digit_fraction": "digit fraction", "punctuation_fraction": "punctuation fraction",
                 "nonascii_fraction": "non-ASCII fraction", "html_body": "HTML part present", "html_only": "HTML only",
                 "attachments": "attachments", "stopwords": "stop words", "urgent_words": "urgency words",
                 "email_addresses": "email addresses", "url_count": "distinct URLs", "unique_url_hosts": "distinct hosts",
                 "ip_url_count": "IP-host URLs", "max_host_dots": "dots in host", "anchor_count": "anchors",
                 "mismatched_anchors": "mismatched anchors", "action_link_text": "action link text",
                 "script_or_form": "script or form"}
MM = 1 / 25.4


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
        "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "axes.grid": False,
        "legend.frameon": False,
    })


def save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.pdf", out / f"{stem}.png"]
    # No creation date, so that a rerun reproduces the files byte for byte.
    meta_pdf = {"Creator": "Matplotlib", "Author": "", "Title": "", "Subject": "", "Keywords": "",
                "CreationDate": None}
    fig.savefig(paths[0], metadata=meta_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[1], metadata={"Software": "Matplotlib"}, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return paths


def _box(ax, x, y, w, h, title, body, face="#f5f4f1"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0.6, edgecolor=MUTED, facecolor=face))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=7, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.048, body, ha="center", va="top", fontsize=6.2, linespacing=1.25,
            color=INK)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, linewidth=0.6,
                                 color=MUTED, shrinkA=0, shrinkB=0))


def workflow(summary: dict, out: Path) -> list[Path]:
    """Figure 1: the executed workflow, with unit counts taken from datasets/corpus_summary.json."""
    style()
    u = summary["units_per_collection"]
    fig = plt.figure(figsize=(158 * MM, 118 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    yr = {c: f"{v['50%']:.0f}" for c, v in summary.get("units_years", {}).items()}
    cols = [("NZ", "Nazario phishing", "annual mailboxes\n2015-2021, 2025", u.get("nazario_recent", 0)),
            ("PP", "Phishing Pot", f"honeypot samples\nmedian year {yr.get('phishing_pot', '')}",
             u.get("phishing_pot", 0)),
            ("SA", "SpamAssassin", f"non-spam parts\nmedian year {yr.get('spamassassin_ham', '')}",
             u.get("spamassassin_ham", 0)),
            ("EN", "Enron", "10% hash sample\nand folder rule", u.get("enron", 0)),
            ("AL", "Apache lists", "six user lists, 2015-2025\nnot used to train detectors", u.get("apache_lists", 0))]
    w, gap = 0.18, (1 - 5 * 0.18) / 6
    for i, (code, name, desc, n) in enumerate(cols):
        x = gap + i * (w + gap)
        face = "#e8f0fb" if i < 2 else ("#eaf6f0" if i < 4 else "#f7efe6")
        _box(ax, x, 0.845, w, 0.14, f"{name} ({code})", f"{desc}\n{n:,} units", face)
        _arrow(ax, x + w / 2, 0.845, x + w / 2, 0.81)
    _box(ax, 0.03, 0.70, 0.94, 0.11, "Safe parsing and screening",
         "message decoding in memory; links never visited; attachments counted, not decoded; one text view per message;\n"
         "included when at least 20 non-space characters remain and the language is identified as English")
    _arrow(ax, 0.5, 0.70, 0.5, 0.665)
    _box(ax, 0.03, 0.555, 0.94, 0.11, "Cross-collection grouping and unit freeze",
         "exact, template and near-duplicate links (Jaccard of word 5-shingles at least 0.8); label-conflict and\n"
         "cross-collection groups removed; one representative per group; five seeded 50/10/20/20 splits per collection")
    _arrow(ax, 0.30, 0.555, 0.30, 0.52)
    _arrow(ax, 0.82, 0.555, 0.82, 0.52)
    _box(ax, 0.03, 0.335, 0.60, 0.185, "Four training pairings (P, L)",
         "NZ-SA (primary), NZ-EN, PP-SA, PP-EN\n"
         "10 variants: logistic, random forest, gradient boosting, MLP,\n"
         "single-token attention control, feature-token Transformer,\n"
         "two ablations, format-only control, TF-IDF logistic\n"
         "2 settings \u00d7 2 class weightings, chosen on tune average precision")
    _box(ax, 0.66, 0.335, 0.31, 0.185, "Collection-identity probes",
         "models trained to tell NZ from PP,\nSA from EN, SA from AL, EN from AL\n(the target is the collection,\nnot phishing)")
    _arrow(ax, 0.33, 0.335, 0.33, 0.30)
    _box(ax, 0.03, 0.19, 0.60, 0.11, "False-alarm thresholds",
         "Neyman-Pearson order statistic on L calibration units only;\n"
         "budgets \u03b1 = 0.1%, 0.5% and 1% with \u03b4 = 0.05; unavailable below the minimum count")
    _arrow(ax, 0.33, 0.19, 0.33, 0.155)
    _box(ax, 0.03, 0.02, 0.60, 0.135, "Scores saved before any metric",
         "in-distribution = P test + L test; phishing shift = all P′ + L test;\n"
         "legitimate shift = P test + all L′; both shift = all P′ + all L′;\n"
         "modern legitimate = P test + all AL; P′ with AL = all P′ + all AL")
    _box(ax, 0.66, 0.02, 0.31, 0.285, "Analysis and verification",
         "average precision, recall at calibrated\nbudgets, realised false-positive rate,\n"
         "paired two-level bootstrap with Holm\nadjustment, permutation importance;\n"
         "an independent verifier recomputes\nthe metrics, thresholds and tests\nfrom the saved scores")
    _arrow(ax, 0.63, 0.09, 0.66, 0.09)
    _arrow(ax, 0.815, 0.335, 0.815, 0.305)
    return save(fig, out, "Figure_1_Workflow")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
