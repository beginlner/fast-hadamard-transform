import torch
from fast_hadamard_transform import fast_hadamard_transform_interface

N = 12
M = 4
DIM = N * (2 ** M)
BATCH = 3 * 4096 * 128
FUNC = getattr(fast_hadamard_transform_interface, "hadamard_transform" + (f"_{N}N" if N > 1 else ""))
I = torch.eye(DIM, device="cuda")
X = FUNC(I, 1.0, "e4m3_pt_simulated", True)
print(X)
Y = FUNC(X, 1.0, "e4m3_pt", True)[0].float()
print(Y)
print(DIM, len(Y.nonzero().tolist()))

X = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="cuda")
B = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="cuda")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for i in range(10):
        FUNC(X, 1.0, "")
    for i in range(10):
        FUNC(X, 1.0, "e4m3_pt", True)
    for i in range(10):
        FUNC(X, 1.0, "e4m3_pt_simulated", True)
    for i in range(10):
        X.clone()
    for i in range(10):
        X @ B
print(prof.key_averages().table(sort_by="cuda_time_total"))


had_12 = torch.tensor([
    [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
    [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
    [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
    [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
    [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
    [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
    [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
    [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
    [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
    [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
    [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
])
had_16 = torch.tensor([
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
    [1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1],
    [1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1],
    [1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1],
    [1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1],
    [1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1],
    [1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1],
    [1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1],
    [1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1],
    [1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1],
    [1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
    [1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1],
    [1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],
    [1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1],
])

had_192 = torch.kron(had_16, had_12)
# print((had_192 @ had_192.T).nonzero().tolist())
