"""Software and algebra checks on hand-written inputs.

These inputs are deterministic test vectors and minimal MIME skeletons without
message content; they are never used as data and produce no results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phishstudy import verify  # noqa: E402
from phishstudy.dedup import group_messages  # noqa: E402
from phishstudy.features import FEATURES, engineered, model_text  # noqa: E402
from phishstudy.models import FeatureTokenTransformer, SingleTokenAttention  # noqa: E402
from phishstudy.parse import parse_message  # noqa: E402
from phishstudy.splits import assign  # noqa: E402
from phishstudy.stats import at_threshold, clopper_pearson, holm, minimum_negatives, np_threshold  # noqa: E402

LABELS = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
SCORES = np.array([.9, .8, .8, .7, .6, .55, .55, .55, .3, .2, .2, .1, .05, .05, .0])


def test_minimum_negatives():
    assert [minimum_negatives(a) for a in (.001, .005, .01)] == [2995, 598, 299]


def test_threshold_unavailable_below_minimum():
    t = np_threshold(np.linspace(0, 1, 298), .01)
    assert t["threshold"] is None and t["status"].startswith("UNAVAILABLE")


def test_threshold_matches_independent_rank():
    neg = (np.arange(3100) * 7919 % 1009) / 1009.0
    for a in (.001, .005, .01):
        t = np_threshold(neg, a)
        k, need = verify.np_rank(len(neg), a, .05)
        assert t["rank"] == k and t["required"] == need
        assert t["threshold"] == float(np.sort(neg)[k - 1])
        assert t["violation_bound"] <= .05


def test_ap_and_auc_match_sklearn_with_ties():
    assert abs(verify.ap(LABELS, SCORES) - average_precision_score(LABELS, SCORES)) < 1e-12
    assert abs(verify.auc(LABELS, SCORES) - roc_auc_score(LABELS, SCORES)) < 1e-12


def test_clopper_pearson_two_implementations():
    for k, n in [(0, 10), (3, 10), (10, 10), (7, 4400), (2990, 3000)]:
        a = clopper_pearson(k, n)
        b = verify.clopper_pearson(k, n)
        assert abs(a[0] - b[0]) < 1e-7 and abs(a[1] - b[1]) < 1e-7
    assert abs(clopper_pearson(3, 10)[0] - beta.ppf(.025, 3, 8)) < 1e-12


def test_confusion_counts_strict_inequality():
    m = at_threshold(LABELS, SCORES, .55)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 2, 7, 3)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(LABELS)


def test_holm():
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])


def test_single_token_attention_is_affine():
    torch.manual_seed(0)
    m = SingleTokenAttention(5, 8).eval()
    x = torch.linspace(-1, 1, 15).reshape(3, 5)
    z = m.project(x).unsqueeze(1)
    _, w = m.attn(z, z, z, need_weights=True)
    assert torch.allclose(w, torch.ones_like(w))
    v = torch.nn.functional.linear(z, m.attn.in_proj_weight[16:], m.attn.in_proj_bias[16:])
    out = m.attn.out_proj(v)
    a, _ = m.attn(z, z, z)
    assert torch.allclose(a, out, atol=1e-6)


def test_feature_token_transformer_shapes_and_tokens():
    m = FeatureTokenTransformer(len(FEATURES), 16).eval()
    x = torch.zeros(4, len(FEATURES))
    assert m.tokens(x).shape == (4, len(FEATURES) + 1, 16)
    assert m(x).shape == (4,)


def test_mime_pruning_and_alternative_choice():
    skeleton = (b"Subject: alpha\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=\"B\"\r\n\r\n"
                b"--B\r\nContent-Type: multipart/alternative; boundary=\"C\"\r\n\r\n"
                b"--C\r\nContent-Type: text/plain\r\n\r\nbeta gamma\r\n"
                b"--C\r\nContent-Type: text/html\r\n\r\n<p>beta <a href=\"http://h.example/x\">delta</a></p>\r\n--C--\r\n"
                b"--B\r\nContent-Type: text/plain\r\nContent-Disposition: attachment; filename=\"a.txt\"\r\n\r\n"
                b"epsilon\r\n--B--\r\n")
    p = parse_message(skeleton)
    assert p["subject"] == "alpha" and p["body"] == "beta gamma"
    assert p["has_html"] and p["has_plain"] and p["attachments"] == 1
    assert p["anchors"] == [("http://h.example/x", "delta")]
    f = engineered(p["subject"], p["body"], p)
    assert f["url_count"] == 1 and f["anchor_count"] == 1 and f["html_only"] == 0


def test_model_text_masks_addresses_and_collector_tokens():
    t = model_text("Hello Jose", "write to phishing@pot or visit http://h.example/x")
    assert "jose" not in t.lower() and "phishing@pot" not in t and "http" not in t
    assert "emailtoken" in t and "urltoken" in t


def test_grouping_exact_template_near():
    tokens = [f"w{i}" for i in range(60)]
    base = " ".join(tokens)
    one_changed = " ".join(tokens[:30] + ["zz"] + tokens[31:])
    one_inserted = " ".join(tokens[:2] + ["9"] + tokens[2:])
    texts = [base, base, one_changed, "unrelated words only here", one_inserted]
    templ = [t.replace("9", "0") for t in texts]
    g, pairs = group_messages(texts, templ, threshold=0.8)
    assert g[0] == g[1] == g[2] == g[4] and g[3] != g[0]


def test_splits_partition_and_determinism():
    units = pd.DataFrame(dict(record_id=[f"r{i:03d}" for i in range(100)],
                              collection=["a"] * 60 + ["b"] * 40))
    s1, s2 = assign(units, [17, 29]), assign(units, [17, 29])
    assert s1.equals(s2)
    for rep in (0, 1):
        part = s1[s1.replicate.eq(rep)]
        assert part.record_id.is_unique and len(part) == 100
        counts = part.groupby(["collection", "split"]).size()
        assert counts[("a", "fit")] == 30 and counts[("b", "test")] == 8


def test_enron_rules_and_automated_senders():
    from phishstudy.corpus import automated_sender, enron_rule
    assert enron_rule("maildir/a-b/sent_items/1.", {"folder": "sent_items"}) == "enron_folder_rule"
    assert enron_rule("maildir/a-b/deleted_items/1.", {"folder": "deleted_items"}) == "enron_folder_rule"
    kept = [p for p in (f"maildir/u/inbox/{i}." for i in range(200)) if enron_rule(p, {"folder": "inbox"}) == ""]
    assert 0 < len(kept) < 60  # about one in ten paths by hash
    assert automated_sender("jira@apache.org") and automated_sender("noreply@example.org")
    assert not automated_sender("lucia@example.org") and not automated_sender("digital.team@example.org")


def test_download_format_checks(tmp_path):
    from phishstudy.acquire import check_format
    good = tmp_path / "a.mbox"
    good.write_bytes(b"From x@y Mon Jan  1 00:00:00 2024\nSubject: a\n\nb\n")
    check_format(good, "mbox")
    page = tmp_path / "b.mbox"
    page.write_bytes(b"<!DOCTYPE HTML><html>blocked</html>")
    with pytest.raises(ValueError):
        check_format(page, "mbox")
    empty = tmp_path / "c.mbox"
    empty.write_bytes(b"")
    check_format(empty, "mbox", allow_empty=True)
    with pytest.raises(ValueError):
        check_format(empty, "mbox")


def test_threshold_flags_strictly_above():
    neg = np.arange(1, 301) / 300.0
    t = np_threshold(neg, 0.01)
    assert at_threshold(np.r_[np.zeros(300, int), [1]], np.r_[neg, [t["threshold"]]], t["threshold"])["tp"] == 0


def test_verifier_close_handles_missing():
    assert verify.close(None, float("nan")) and not verify.close(None, 0.0)
    with pytest.raises(verify.AuditError):
        verify.check(False, "x")
