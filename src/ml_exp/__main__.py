import torch
import torch.nn.functional as F

from .data import Dataset
from .rnn.rnn import RNN

epoch = 10_000
lr = 5e-4
state_size = 512
vocab_embedding_size = 128
hidden_size = 256
chain = 10
batch_size = 12
training_leng = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

data = Dataset("./data/data.txt", device=device)
model = RNN(
    state_size,
    data.vocab_size,
    vocab_embedding_size=vocab_embedding_size,
    hidden_size=hidden_size,
    rnnchain_count=chain,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


model.train()
for i in range(epoch):
    state = model.newstate(batch_size, device=device)
    x, y = data.get_batch("train", training_leng, batch_size)
    logits, state, loss = model(x, state, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"{i}th iteration: {loss}")


state = model.newstate(1, device=device)
x, y = data.get_batch("train", 32, 1)
logits, state = model(x, state)

probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)

list = idx_next.reshape(32).tolist()
print(data.vocab_size)
print(list)
print(data.decode(list))
