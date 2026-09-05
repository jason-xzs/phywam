import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import infer_mope
from dataset.robotwin_event_dataset import RobotWinEventBlockDataset
from run_jepa_pretraining import get_model, load_pretrained_checkpoint


class EvalTransform:
    """Deterministic center-crop transform with a full-visible token mask."""

    def __init__(self, input_size=224, num_patches=1568):
        self.frame_transform = infer_mope.build_inference_transform(input_size)
        self.num_patches = num_patches

    def __call__(self, sample):
        images, _ = sample
        process_data = torch.cat(
            [self.frame_transform(image) for image in images], dim=0)
        visible = torch.zeros(self.num_patches, dtype=torch.bool)
        return process_data, visible, visible.clone()


def parse_args():
    parser = argparse.ArgumentParser(
        'Evaluate new-architecture MoPE on held-out RobotWin v39 segments.')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--datasets-root', required=True)
    parser.add_argument('--event-label-path', required=True)
    parser.add_argument('--physics-soft-path', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (
        args.ckpt,
        args.datasets_root,
        args.event_label_path,
        args.physics_soft_path,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    model_args = SimpleNamespace(
        model='pretrain_mope_jepa_base_patch16_224',
        use_mope=True,
        num_physics_experts=17,
        num_general_experts=10,
        num_shared_experts=4,
        candidate_k=5,
        gate_threshold=0.0,
        gate_hidden=0,
        gate_layers=2,
        gate_dims='',
        num_event_classes=28,
        drop_path=0.0,
        num_frames=16,
        tubelet_size=2,
        with_checkpoint=False,
        event_label_path=args.event_label_path,
        stage1_train_classifier=False,
        stage2_train_experts=False,
        freeze_encoder_except_moe=True,
    )
    model = get_model(model_args)
    load_pretrained_checkpoint(model, args.ckpt)
    if model.event_head is None:
        raise RuntimeError('Checkpoint/model has no v39 event head')
    model.encoder.enable_general = False

    transform = EvalTransform(
        input_size=224,
        num_patches=model.encoder.patch_embed.num_patches,
    )
    dataset = RobotWinEventBlockDataset(
        datasets_root=args.datasets_root,
        event_label_path=args.event_label_path,
        transform=transform,
        num_frames=16,
        physics_soft_path=args.physics_soft_path,
        has_physics_label=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    event_correct = 0
    physics_correct = 0
    total = 0
    per_event_total = Counter()
    per_event_correct = Counter()

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            physics_soft = batch[4].to(device, non_blocking=True)
            event_label = batch[5].to(device, non_blocking=True)
            full_mask = torch.zeros(
                images.size(0),
                model.encoder.patch_embed.num_patches,
                dtype=torch.bool,
                device=device,
            )
            x_vis = model.encoder(images, full_mask)
            event_logits = model.event_head(x_vis.mean(dim=1))
            event_pred = event_logits.argmax(dim=-1)
            physics_scores = model.encoder._cls_scores[:, :17]
            physics_pred = physics_scores.argmax(dim=-1)
            physics_target = physics_soft[:, :17].argmax(dim=-1)

            matches = event_pred.eq(event_label)
            event_correct += int(matches.sum().item())
            physics_correct += int(
                physics_pred.eq(physics_target).sum().item())
            for label, correct in zip(
                    event_label.cpu().tolist(), matches.cpu().tolist()):
                name = dataset.event_vocabulary[int(label)]
                per_event_total[name] += 1
                per_event_correct[name] += int(correct)
            total += images.size(0)
            if args.max_samples > 0 and total >= args.max_samples:
                break

    metrics = {
        'checkpoint': str(Path(args.ckpt).resolve()),
        'num_samples': total,
        'event_top1': event_correct / max(total, 1),
        'physics_top1': physics_correct / max(total, 1),
        'enable_general': False,
        'per_event_top1': {
            name: per_event_correct[name] / count
            for name, count in sorted(per_event_total.items())
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
