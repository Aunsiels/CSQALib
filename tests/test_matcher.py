from utils.matcher import Matcher


def test_matcher():
    concept_ids = ["girl", "high_school", "man"]
    sentence = "the high school girl"
    
    matcher = Matcher(concept_ids)
    assert matcher(sentence) == {"girl", "high_school"}
    