class CharTokenizer:

    def __init__(self, text=None, stoi=None):
        if stoi is None:
            if text is None:
                raise ValueError("Either text or stoi must be provided.")
            chars = sorted(set(text))
            self.stoi = {c: i for i, c in enumerate(chars)}
        else:
            self.stoi = dict(stoi)

        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)
