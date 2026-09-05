import time
import torch
import multiprocessing as mp
import argparse


def occupy_gpu(
    device_id: int,
    target_gib: float = 120.0,
    block_gib: float = 1.0,
    matrix_size: int = 16384,
    inner_iters: int = 50,
):
    """
    占满显存 + 拉满算力。
    """
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    bytes_per_block = int(block_gib * (1024 ** 3))
    numel_per_block = bytes_per_block // 4  # float32

    print(f"[GPU {device_id}] Target ~{target_gib} GiB, block {block_gib} GiB")

    blocks = []
    allocated_gib = 0.0
    try:
        # 先把显存占满，但要给 matmul 的工作矩阵留出空间
        reserve_gib = block_gib * 4  # 给计算矩阵留余量
        while allocated_gib < target_gib - reserve_gib:
            try:
                t = torch.empty(numel_per_block, device=device, dtype=torch.float32)
                t.fill_(0)
                blocks.append(t)
                allocated_gib += block_gib
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[GPU {device_id}] OOM at {allocated_gib:.1f} GiB.")
                    torch.cuda.empty_cache()
                    break
                raise
        print(f"[GPU {device_id}] Allocated ~{allocated_gib:.1f} GiB. Burning compute...")

        # 大矩阵拉满算力。可改 dtype=torch.float16/bfloat16 走 Tensor Core 拉满更多
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float32)

        print(f"[GPU {device_id}] Burning. Ctrl+C to stop.")
        while True:
            for _ in range(inner_iters):
                torch.mm(a, b, out=c)  # 无空隙连续 matmul，复用 c 不申显存
            torch.cuda.synchronize()
    except KeyboardInterrupt:
        print(f"[GPU {device_id}] Stopped.")


def occupy_all_gpus(gpus, target_gib, block_gib, matrix_size, inner_iters):
    print(f"Will occupy GPUs: {gpus}, target {target_gib} GiB each")
    mp.set_start_method("spawn", force=True)
    procs = []
    for dev_id in gpus:
        p = mp.Process(
            target=occupy_gpu,
            args=(dev_id,),
            kwargs=dict(
                target_gib=target_gib,
                block_gib=block_gib,
                matrix_size=matrix_size,
                inner_iters=inner_iters,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="4,5,6,7")
    parser.add_argument("--target-gib", type=float, default=120.0)
    parser.add_argument("--block-gib", type=float, default=1.0)
    parser.add_argument("--matrix-size", type=int, default=16384)
    parser.add_argument("--inner-iters", type=int, default=50)
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    occupy_all_gpus(
        gpus=gpus,
        target_gib=args.target_gib,
        block_gib=args.block_gib,
        matrix_size=args.matrix_size,
        inner_iters=args.inner_iters,
    )