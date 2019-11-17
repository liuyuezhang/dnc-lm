import torch
import dnc

device = torch.device("cuda")

rnn = dnc.SDNC(
    input_size=64,
    hidden_size=128,
    rnn_type='lstm',
    num_layers=4,
    nr_cells=100,
    cell_size=32,
    read_heads=4,
    sparse_reads=4,
    batch_first=True,
    gpu_id=0
).to(device)

(controller_hidden, memory, read_vectors) = (None, None, None)

torch.autograd.set_detect_anomaly(True)

output, (controller_hidden, memory, read_vectors) = \
    rnn(torch.randn(10, 4, 64).to(device), (controller_hidden, memory, read_vectors), reset_experience=True)
loss = torch.sum(output)
loss.backward()

optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)