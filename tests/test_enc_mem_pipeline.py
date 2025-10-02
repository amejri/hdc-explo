# test_enc_mem_pipeline.py
import numpy as np
import pytest

from hdc_project.encoder import m4, pipeline as enc_pipeline
from hdc_project.encoder.mem import pipeline as mem_pipeline

# ---------- helpers ----------
def content_signature_from_Xseq(X_seq, majority: str = "strict"):
    if not X_seq:
        raise ValueError("X_seq vide")
    S = np.zeros((X_seq[0].shape[0],), dtype=np.int32)
    for x in X_seq:
        S += x.astype(np.int32, copy=False)
    # seuil strict (tie -> +1)
    return np.where(S >= 0, 1, -1).astype(np.int8, copy=False)

def span_signatures_from_trace(X_seq, win: int = 12, stride: int = 6, majority: str = "strict"):
    if not X_seq:
        return []
    T = len(X_seq)
    out = []
    if T <= win:
        out.append(content_signature_from_Xseq(X_seq, majority))
        return out
    for start in range(0, T - win + 1, max(1, stride)):
        stop = start + win
        out.append(content_signature_from_Xseq(X_seq[start:stop], majority))
    return out

def build_mem_pairs_from_encoded(encoded_en, encoded_fr, win=12, stride=6, majority="strict", max_pairs=None):
    pairs = []
    N = min(len(encoded_en), len(encoded_fr))
    for i in range(N):
        X_en = encoded_en[i]["X_seq"]
        X_fr = encoded_fr[i]["X_seq"]
        spans_en = span_signatures_from_trace(X_en, win=win, stride=stride, majority=majority)
        spans_fr = span_signatures_from_trace(X_fr, win=win, stride=stride, majority=majority)
        L = min(len(spans_en), len(spans_fr))
        for t in range(L):
            pairs.append((
                spans_en[t].astype(np.int8, copy=False),
                spans_fr[t].astype(np.int8, copy=False),
            ))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs

def mean_intra_bucket_coherence(mem, pairs, buckets, D):
    """Moyenne des <H_c, Z_fr>/D sur tous les (Z_en, Z_fr) affectés à c."""
    H = mem.H.astype(np.int32, copy=False)
    sims = []
    for (z_en, z_fr), c in zip(pairs, buckets):
        sims.append(float(np.dot(H[c], z_fr.astype(np.int32, copy=False)) / D))
    return float(np.mean(sims)) if sims else 0.0

# ---------- test principal ----------
@pytest.mark.parametrize("D,n,B,K,win,stride,probe_count_min", [
    (4096, 5, 128, 16, 12, 6, 100),
])
def test_enc_mem_e2e(D, n, B, K, win, stride, probe_count_min):
    # --- 0) données (petit corpus pour aller vite)
    ens_raw = [
        "hyperdimensional computing is fun",
        "vector symbolic architectures are powerful",
        "encoding words into hyperspace",
        "memory augmented networks love clean data",
        "alignment of spans matters in retrieval",
        "binding keys should not leak into content",
        "statistics of buckets should be reasonably balanced",
        "random seeds must make tests deterministic",
        "ngrams superpose and then majority wins",
        "french and english streams are separate lexica",
    ] * 50  # 500 phrases
    frs_raw = [
        "le calcul hyperdimensionnel est amusant",
        "les architectures symboliques vectorielles sont puissantes",
        "encoder des mots dans l'hyperspace",
        "les réseaux augmentés de mémoire aiment les données propres",
        "l'alignement des fenêtres est crucial",
        "la clé de segment ne doit pas fuir dans le contenu",
        "les seaux de hachage doivent être assez équilibrés",
        "les graines aléatoires rendent les tests déterministes",
        "les ngrams se superposent puis la majorité tranche",
        "les lexiques français et anglais sont séparés",
    ] * 50

    enc_sample_size = min(len(ens_raw), 500)
    ens_sample = ens_raw[:enc_sample_size]
    frs_sample = frs_raw[:enc_sample_size]

    # --- 1) ENC (mêmes seeds & permutation partagée)
    rng = np.random.default_rng(123)
    Lex_en = m4.M4_LexEN_new(seed=1, D=D)
    Lex_fr = m4.M4_LexEN_new(seed=2, D=D)
    pi = rng.permutation(D).astype(np.int64)

    encoded_en = enc_pipeline.encode_corpus_ENC(ens_sample, Lex_en, pi, D, n, seg_seed0=999)
    encoded_fr = enc_pipeline.encode_corpus_ENC(frs_sample, Lex_fr, pi, D, n, seg_seed0=1999)

    # Sanity ENC
    assert len(encoded_en) == enc_sample_size
    assert encoded_en[0]["H"].shape == (D,)
    # Sim inter-segments ~petite (ordre 0.005–0.02 typiquement)
    inter_seg = enc_pipeline.inter_segment_similarity([e["H"] for e in encoded_en])
    assert 0.0 <= inter_seg < 0.05

    # --- 2) paires MEM de contenu (sans K_s)
    pairs_mem = build_mem_pairs_from_encoded(encoded_en, encoded_fr, win=win, stride=stride, majority="strict")
    assert len(pairs_mem) > B * 10, "pas assez d’exemples par bucket"

    # --- 3) MEM: instanciation & entraînement
    cfg = mem_pipeline.MemConfig(D=D, B=B, k=K, seed_lsh=10, seed_gmem=11)
    comp = mem_pipeline.make_mem_pipeline(cfg)
    mem_pipeline.train_one_pass_MEM(comp, pairs_mem)

    # Comptes: somme == nb paires
    assert int(comp.mem.n.sum()) == len(pairs_mem)
    # Utilisation: au moins ~60% des buckets touchés
    used = int((comp.mem.n > 0).sum())
    assert used >= int(0.6 * B)

    # Skew raisonnable (p99/p50 pas monstrueux)
    counts = np.sort(comp.mem.n[comp.mem.n > 0])
    p50 = np.median(counts)
    p99 = counts[int(0.99 * (counts.size - 1))]
    assert p99 / max(p50, 1) < 8.0, "distribution de buckets trop dégénérée"

    # --- 4) Probe: top-1 similarity positive (signal)
    probe_count = min(probe_count_min, len(pairs_mem))
    sims = []
    buckets = []
    for (z_en, z_fr) in pairs_mem[:probe_count]:
        c_star, _ = mem_pipeline.infer_map_top1(comp, z_en)
        buckets.append(c_star)
        proto = comp.mem.H[c_star].astype(np.int32, copy=False)
        sims.append(float(np.dot(proto, z_fr.astype(np.int32, copy=False)) / D))

    mean_sim = float(np.mean(sims)) if sims else 0.0
    med_sim = float(np.median(sims)) if sims else 0.0

    # seuils prudents (attendu ~0.20–0.30 selon win/stride/D)
    assert mean_sim >= 0.12, f"mean similarity trop faible: {mean_sim:.3f}"
    assert med_sim >= 0.12, f"median similarity trop faible: {med_sim:.3f}"

    # --- 5) Cohérence intra-bucket moyenne (pondérée par la fréquence naturelle)
    coh = mean_intra_bucket_coherence(comp.mem, pairs_mem[:probe_count], buckets, D)
    assert coh >= 0.10, f"cohérence intra-bucket trop faible: {coh:.3f}"

    # --- 6) Vérifs structurelles des vecteurs
    # Tous les vecteurs utilisés doivent être en {-1,+1}
    def is_pm1(v): 
        u = np.unique(v)
        return u.size == 2 and set(u.tolist()) == {-1, 1}
    # on teste quelques exemples
    for k in [0, min(1, len(pairs_mem)-1), min(2, len(pairs_mem)-1)]:
        ze, zf = pairs_mem[k]
        assert is_pm1(ze) and is_pm1(zf)
    assert is_pm1(comp.mem.H[0])