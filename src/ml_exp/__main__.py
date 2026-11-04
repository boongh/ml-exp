import torch
import torch.nn.functional as F

from .data import Dataset
from .rnn.rnn import RNN

epoch = 10_000
lr = 5e-5
state_size = 4096
vocab_embedding_size = 128
hidden_size = 512
chain = 20

data = Dataset("./data/data.txt")
model = RNN(
    state_size,
    data.vocab_size,
    vocab_embedding_size=vocab_embedding_size,
    hidden_size=hidden_size,
    rnnchain_count=chain,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


model.train()
for i in range(epoch):
    state = model.newstate()
    x, y = data.get_batch("train", 128, 1)
    logits, state, loss = model(x, state, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"{i}th iteration: {loss}")


state = model.newstate()
x, y = data.get_batch("train", 32, 1)
logits, state = model(x, state)

probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)

list = idx_next.reshape(32).tolist()
print(data.vocab_size)
print(list)
print(data.decode(list))
