import torch
import torch.nn.functional as F

from .data import Dataset
from .rnn.rnn import RNNLangModel

epoch = 10_000
lr = 1e-4
dropout = 0.4
state_size = 512 + 256
vocab_embedding_size = 256
hidden_size = 128
chain = 12
batch_size = 12
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
    logits, state = model.generate(x, state, 128)

    probs = F.softmax(logits, dim=-1)
    bprobs, tprobs, nprobs = probs.size()
    idx_next = (
        torch.multinomial(probs.view(bprobs * tprobs, nprobs), num_samples=numsamples)
        .view(bprobs, tprobs, numsamples)
        .squeeze()
    )

    list = idx_next.tolist()

    print(idx_next.size())

    if numsamples == 1:
        for i in list:
            print(data.decode(i))

    model.train()


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
        state = model.newstate(batch_size, device=device)
        x, y = data.get_batch("eval", training_leng, batch_size)
        logits, state, eloss = model(x, state, y)
        print(f"{i}th iteration: {loss} and {eloss} eval loss")

model.eval()

test_generation()
