from preprocess import make_idxs


def label_original_tokens(original_tokenss, y_pred):
    labeled_tokenss = []
    idxs = make_idxs(original_tokenss)
    for i, tokens in enumerate(original_tokenss):
        labels_pred = ["O"] * len(tokens)
        nums = idxs[i]
        for x, n in enumerate(nums):
            labels_pred[int(n)] = y_pred[i][x]
        labeled_tokens = list(zip(tokens, labels_pred))
        labeled_tokenss.append(labeled_tokens)
    return labeled_tokenss


def get_all_bskill_index(labeled_tokens):
    bskill_index = [
        i
        for i, (token, labels_pred) in enumerate(labeled_tokens)
        if labels_pred == "B-SKILL"
    ]
    bskill_index.append(len(labeled_tokens))
    return [
        (bskill_index[i], bskill_index[i + 1]) for i in range(len(bskill_index) - 1)
    ]


def get_all_skill_index(labeled_tokens):
    idx_pairs = get_all_bskill_index(labeled_tokens)
    skill_index_pairs = []
    for m, n in idx_pairs:
        k = m + 1
        for i in range(m, n):
            if labeled_tokens[i][1] == "I-SKILL":
                k = i + 1
        skill_index_pairs.append((m, k))
    return skill_index_pairs


def export_html(labeled_tokens):
    idx_pairs = get_all_skill_index(labeled_tokens)
    labeled_original_tokens = labeled_tokens
    highlighted_text = ""
    for m, n in idx_pairs:
        for i in range(m, n):
            labeled_original_tokens[i] = (labeled_tokens[i][0], "SKILL")
    skills = []
    for token, label in labeled_original_tokens:
        if label == "SKILL":
            skills.append(token)
        else:
            if skills:
                highlighted_text += "<mark>" + "".join(skills) + "</mark>"
            highlighted_text += token
            skills = []
    if skills:
        highlighted_text += "<mark>" + "".join(skills) + "</mark>"
    highlighted_text = "<br />".join(highlighted_text.split("\n"))
    return highlighted_text
