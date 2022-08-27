import json


def load_csqa(path):
    def fmt_one(sample):
        x = sample["question"]
        return {
            "question": x["stem"],
            "choices": x["choices"],
            "label": ord(sample["answerKey"]) - ord("A"),
        }

    return [fmt_one(json.loads(line)) for line in open(path)]
