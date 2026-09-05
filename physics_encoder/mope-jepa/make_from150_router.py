"""
从 150.pth 提取 block11 的 token_router 权重，塞进新版 phys_router.gate，
存成新 ckpt 用于推理对比。
  - backbone: 加载 VideoMAEv2 预训练权重（和 150 训练时一致）
  - phys_router.gate: 用 150.pth 的 blocks.11 token_router 权重
其余（专家等）随机初始化（推理分类只用 phys_router 的 token_scores.mean，不影响）。
"""
import argparse, torch, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import models  # noqa
from timm.models import create_model
from collections import OrderedDict
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt150', default='/data2/mope-jepa/output/mope_jepa_wisa7k_vitb_freeze/checkpoint-150.pth')
    ap.add_argument('--videomae', default='/data2/mope-jepa/pretrained/videomaev2_base.pth')
    ap.add_argument('--src_block', type=int, default=11, help='用150.pth哪一层的token_router(8-11)')
    ap.add_argument(
        '--out',
        default=str(ROOT / 'output/stage1_classifier/from150_router.pth'))
    args = ap.parse_args()

    # 1. 建新版模型
    model = create_model('pretrain_mope_jepa_base_patch16_224', pretrained=False,
                         all_frames=16, tubelet_size=2,
                         num_physics_experts=17, num_general_experts=10,
                         num_shared_experts=4)

    # 2. 加载 VideoMAEv2 backbone（key 映射同训练脚本）
    def _safe_load(p):
        try:
            from torch.serialization import safe_globals
            with safe_globals([np._core.multiarray.scalar]):
                return torch.load(p, map_location='cpu', weights_only=False)
        except Exception:
            return torch.load(p, map_location='cpu', weights_only=False)

    vm = _safe_load(args.videomae)
    vm = vm.get('model', vm.get('module', vm))
    new_vm = OrderedDict()
    for k, v in vm.items():
        if k.startswith('backbone.'):
            new_vm['encoder.' + k[len('backbone.'):]] = v
        elif k.startswith('encoder.'):
            new_vm[k] = v
        else:
            new_vm['encoder.' + k] = v
    msg = model.load_state_dict(new_vm, strict=False)
    print(f"[backbone] loaded VideoMAEv2, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    # 3. 从 150.pth 取 block{src} 的 token_router 权重
    ck = _safe_load(args.ckpt150)
    sd = ck.get('model', ck.get('module', ck))
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    src_key = f'encoder.blocks.{args.src_block}.mlp.sparse_moe.token_router.weight'
    if src_key not in sd:
        print(f"ERROR: {src_key} not in 150.pth"); return
    router_w = sd[src_key]   # [17,768]
    print(f"[router] 取 150.pth 的 {src_key}, shape={tuple(router_w.shape)}")

    # 4. 塞进新版 phys_router.gate
    with torch.no_grad():
        model.encoder.global_router.phys_router.gate.weight.copy_(router_w)
    print(f"[router] 已写入 global_router.phys_router.gate.weight")

    # 5. 存 ckpt
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, args.out)
    print(f"[save] {args.out}")
    print("\n推理命令：")
    print(f"python infer_mope.py --ckpt {args.out} \\")
    print(f"    --video_dir /data2/mope-jepa/datasets/test/combustion/ \\")
    print(f"    --save_dir /tmp/from150/ --use_mope --show_cls --gate_threshold 0.0")

if __name__ == '__main__':
    main()
