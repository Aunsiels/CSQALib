from src.concept_embedder import LM_Embedder


def test_lm_embedder():
    cids = ["in_house", "test"]
    embedder = LM_Embedder()
    out = embedder(cids)
    assert out.shape == (2, 384)
