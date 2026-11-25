import torch
import torch.distributed as dist
import os

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

print(f"[Rank {rank}] Initializing process group", flush=True)

dist.init_process_group(backend="nccl")

for i in range(10):
    tensor = torch.tensor([rank + 1.0], device=device)

    print(f"[Rank {rank}] Before all_reduce {i}", flush=True)

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(
        f"[Rank {rank}] After all_reduce {i}: {tensor.item()}",
        flush=True
    )

dist.destroy_process_group()
print(f"[Rank {rank}] Finished", flush=True)