from collections import Counter

from hdc_project.infer_opus import _prepare_lm_resources


def test_fallback_includes_heldout_tokens():
    freq_train = Counter({"bonjour": 3, "salut": 1})
    bigram_train = {"bonjour": Counter({"salut": 1})}
    lexicon_sequences = [
        ["hello", "world"],
        ["__sent_marker_0", "hola", "amigo"],
    ]

    freq_lm, bigrams_lm, fallback_vocab, global_vocab, used_fallback = _prepare_lm_resources(
        freq_train,
        bigram_train,
        lexicon_sequences,
    )

    assert not used_fallback
    assert "hello" in fallback_vocab  # token absent from training set
    assert "hola" in freq_lm
    assert "bonjour" in global_vocab  # training token preserved
    assert bigrams_lm["hola"]["amigo"] == 1


def test_fallback_reuses_train_when_lexicon_empty():
    freq_train = Counter({"salut": 2})
    bigram_train = {"salut": Counter({"salut": 1})}

    freq_lm, bigrams_lm, fallback_vocab, global_vocab, used_fallback = _prepare_lm_resources(
        freq_train,
        bigram_train,
        lexicon_sequences=[],
    )

    assert used_fallback
    assert "salut" in fallback_vocab
    assert "salut" in global_vocab
    assert freq_lm["salut"] == 2
    assert bigrams_lm["salut"]["salut"] == 1
