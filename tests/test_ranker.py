from utils.ranker import Ranker

def test_ranker():
    context = "I am eating soup"
    concept_ids = ["soup", "apple", "run"]
    ranker = Ranker()
    scores = ranker(context, concept_ids)
    assert scores.shape == (3,)
    print('ranker scores:', scores)  # [0.79874504 0.1402663  0.28472954]