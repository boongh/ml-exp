import torch
import torch.nn.functional as F

from .data import Dataset
from .rnn.rnn import RNNLangModel

epoch = 5_000
lr = 1e-4
dropout = 0.2
state_size = 1024
vocab_embedding_size = 256
hidden_size = 512
chain = 12
batch_size = 128
training_leng = 48
numsamples = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

data = Dataset("./data/data.txt", device=device)
model = RNNLangModel(
    state_size,
    data.vocab_size,
    vocab_embedding_size=vocab_embedding_size,
    hidden_size=hidden_size,
    rnnchain_count=chain,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def test_generation():
    model.eval()

    x, _ = data.get_batch("eval", training_leng, batch_size)

    state = model.newstate(batch_size, device=device)
    ids, state = model.generate(x, state, 128)

    print(ids.size())
    list = ids.tolist()

    if numsamples == 1:
        for id, i in enumerate(list):
            print(f"Text {id}: '{data.decode(i)}'")

    model.train()


with torch.no_grad():
    test_generation()

model.train()

for i in range(epoch):
    state = model.newstate(batch_size, device=device)
    x, y = data.get_batch("train", training_leng, batch_size)
    logits, state, loss = model(x, state, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        model.eval()
        with torch.no_grad():
            state = model.newstate(batch_size, device=device)
            x, y = data.get_batch("eval", training_leng, batch_size)
            logits, state, eloss = model(x, state, y)
            print(f"{i}th iteration: {loss} and {eloss} eval loss")
        model.train()

model.eval()

with torch.no_grad():
    test_generation()
