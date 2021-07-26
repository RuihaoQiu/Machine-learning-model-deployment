import re
import numpy as np
from nltk import pos_tag

regex_specials = re.compile(r"(\W)")
regex_words = re.compile(r"[\w&.+#]+")


def origin_tokenize(text: str):
    original_tokens = regex_specials.split(text)
    return original_tokens


def make_idxs(original_tokenss):
    idxs = []
    for original_tokens in original_tokenss:
        idx, _ = np.transpose(
            [[i, t] for i, t in enumerate(original_tokens) if regex_words.match(t)]
        )
        idxs.append(idx)
    return idxs


def make_poss(original_tokenss):
    data_poss = []
    for original_tokens in original_tokenss:
        _, tokens = np.transpose(
            [[i, t] for i, t in enumerate(original_tokens) if regex_words.match(t)]
        )
        data_pos = pos_tag(tokens)
        data_poss.append(data_pos)
    return data_poss


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                "-1:postag": postag1,
                "-1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:postag": postag1,
                "+1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
