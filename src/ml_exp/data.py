import torch

class Dataset:
    def __init__(self, data_set, train=0.9, device="cpu"):
        print("Dataset loading...")
        self.device = device
        self.data_set = data_set
        self.text = open(data_set, "r", encoding="utf-8").read() # Limit to first N character for sample purposes. Remove the [:10000] to use the full dataset.
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)
        
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long, device=device)  

        # data.py
        split_idx = int(train * len(self.data))
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

        print(f"Dataset loaded from {data_set}.\n"
              f"Vocabulary size: {self.vocab_size}.\n"
              f"Vocabulary sample: {self.vocab[:10]}\n"
              f"Training data size: {len(self.train_data)}\n"
              f"Validation data size: {len(self.val_data)}")

    def encode(self, text):
        return [self.vocab.index(c) for c in text]

    def decode(self, indices):
        return "".join([self.vocab[i] for i in indices])

    def get_batch(self, type, block_size=16, batch_size=16):
        if type == "train":
            data = self.train_data
        else:
            data = self.val_data
        
        offsets = torch.randint(
            0,
            len(data) - block_size - 1,
            (batch_size,)
        )

        x = torch.stack([
            data[i:i + block_size]
            for i in offsets
        ])

        y = torch.stack([
            data[i + 1:i + block_size + 1]
            for i in offsets
        ])

        return x, y