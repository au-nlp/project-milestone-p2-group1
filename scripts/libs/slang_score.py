
def slang_score(pred, slang_set):
    tokens = pred.split()
    slang_hits = sum(1 for t in tokens if t in slang_set)
    return slang_hits / max(1, len(tokens))  # normalized frequency