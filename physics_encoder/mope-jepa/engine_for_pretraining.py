# --------------------------------------------------------
# engine_for_pretraining.py  ── MoPE-JEPA 版本
#
# 改动：
#   1. MAE loss → JEPA loss（MSE in latent space）
#   2. 新增 SIGReg loss 防 collapse
#   3. 删掉 decode_masked_pos / images_patch / labels 相关逻辑
#   4. model forward 签名：(images, mask, physics_label, physics_label_soft)
# --------------------------------------------------------
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

import utils
from models.sigreg import SIGReg


# ── SIGReg 单例（避免每步重建）────────────────────────────────────────────
_sigreg = None

def get_sigreg(device, knots=17, num_proj=1024):
    global _sigreg
    if _sigreg is None:
        _sigreg = SIGReg(knots=knots, num_proj=num_proj).to(device)
    return _sigreg


def compute_event_loss(
    event_logits: torch.Tensor,
    event_label: torch.Tensor,
    event_score: Optional[torch.Tensor] = None,
    event_label_soft: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Confidence-weighted hard/soft event classification loss."""
    if event_label_soft is not None:
        target = event_label_soft.to(
            dtype=event_logits.dtype, device=event_logits.device)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        loss_raw = -(target * F.log_softmax(event_logits, dim=-1)).sum(dim=-1)
    else:
        loss_raw = F.cross_entropy(
            event_logits, event_label, reduction='none')
    if event_score is None:
        return loss_raw.mean()
    weights = event_score.to(
        dtype=event_logits.dtype, device=event_logits.device)
    weights = weights.clamp(min=0.10, max=1.0)
    return (loss_raw * weights).sum() / weights.sum().clamp_min(1e-6)


# ── 主训练函数 ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    patch_size: int = 16,          # 保留参数签名兼容性，JEPA 不用
    normlize_target: bool = True,  # 同上
    log_writer=None,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    moe_balance_weight: float = 0.01,
    has_physics_label:  bool  = False,
    has_event_label:    bool  = False,
    physics_cls_weight: float = 1.0,
    event_loss_weight:  float = 1.0,
    event_head_only:    bool  = False,
    sigreg_weight:      float = 0.1,
    stage1:             bool  = False,   # 阶段一：只训分类器，仅用 router_loss
    stage2:             bool  = False,   # 阶段二：训专家+通用router，jepa+sigreg+balance
):
    model.train()
    sigreg = get_sigreg(device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'jepa_loss',    utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter(
        'sigreg_loss',  utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter(
        'balance_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter(
        'physics_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter(
        'general_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter(
        'event_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter(
        'event_acc', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    if num_training_steps_per_epoch is not None and num_training_steps_per_epoch > 0:
        print_freq = min(print_freq, num_training_steps_per_epoch)

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        if num_training_steps_per_epoch is not None and step >= num_training_steps_per_epoch:
            break

        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = \
                        lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None \
                        and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # ── batch 解包（JEPA 不需要 decode_masked_pos）────────────────
        physics_label_soft = None
        event_label = None
        event_score = None
        event_label_soft = None
        if has_event_label and has_physics_label and len(batch) >= 8:
            (images, bool_masked_pos, _, physics_label, physics_label_soft,
             event_label, event_score, event_label_soft) = batch[:8]
            physics_label = physics_label.to(device, non_blocking=True)
            physics_label_soft = physics_label_soft.to(
                device, non_blocking=True, dtype=torch.float32)
            event_label = event_label.to(device, non_blocking=True)
            event_score = event_score.to(
                device, non_blocking=True, dtype=torch.float32)
            event_label_soft = event_label_soft.to(
                device, non_blocking=True, dtype=torch.float32)
        elif has_physics_label and len(batch) >= 5:
            images, bool_masked_pos, _, physics_label, physics_label_soft = (
                batch[0], batch[1], batch[2], batch[3], batch[4])
            physics_label = physics_label.to(device, non_blocking=True)
            physics_label_soft = physics_label_soft.to(
                device, non_blocking=True, dtype=torch.float32)
        elif has_physics_label and len(batch) >= 4:
            images, bool_masked_pos, _, physics_label = \
                batch[0], batch[1], batch[2], batch[3]
            physics_label = physics_label.to(device, non_blocking=True)
        else:
            images, bool_masked_pos, _ = batch[0], batch[1], batch[2]
            physics_label = None

        images          = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)

        encoder = getattr(getattr(model, 'module', model), 'encoder', None)

        def _calc_loss(pred, target, event_logits=None):
            # 1. JEPA loss：latent 空间 MSE
            jepa_loss = F.mse_loss(pred, target)

            # 2. SIGReg：context encoder 的 x_vis，防 collapse
            sigreg_loss = pred.new_zeros(())
            if encoder is not None:
                x_vis = getattr(encoder, '_last_x_vis', None)
                if x_vis is not None:
                    # SIGReg 要求输入 (T, B, D)，x_vis 是 (B, N, D)
                    # 把 N 当作 T 传入
                    sigreg_loss = sigreg(x_vis.transpose(0, 1))

            # 3. Balance loss
            balance_loss = pred.new_zeros(())
            if encoder is not None:
                bl = getattr(encoder, '_balance_loss', None)
                if bl is not None:
                    balance_loss = bl

            # 4. 路由监督：physics_loss（物理软标签）+ general_loss（二分类门控）
            #    各 0.5 权重，梯度路径完全分离
            physics_loss = pred.new_zeros(())
            general_loss = pred.new_zeros(())
            if encoder is not None:
                pl = getattr(encoder, '_physics_loss', None)
                if pl is not None:
                    physics_loss = pl
                gl = getattr(encoder, '_general_loss', None)
                if gl is not None:
                    general_loss = gl
            event_loss = pred.new_zeros(())
            event_acc = pred.new_zeros(())
            if has_event_label:
                if event_logits is None or event_label is None:
                    raise RuntimeError(
                        'Event labels are enabled but model returned no event logits')
                event_loss = compute_event_loss(
                    event_logits,
                    event_label,
                    event_score=event_score,
                    event_label_soft=event_label_soft,
                )
                event_acc = (
                    event_logits.argmax(dim=-1) == event_label
                ).to(dtype=pred.dtype).mean()

            total = (
                jepa_loss
                + sigreg_weight * sigreg_loss
                + moe_balance_weight * balance_loss
                + physics_cls_weight * physics_loss
                + general_loss
                + event_loss_weight * event_loss
            )

            if stage1:
                total = physics_loss + general_loss
            return (total, jepa_loss, sigreg_loss, balance_loss,
                    physics_loss, general_loss, event_loss, event_acc)

        autocast_ctx = (
            torch.amp.autocast('cuda')
            if hasattr(torch, 'amp') else torch.cuda.amp.autocast())

        if event_head_only:
            full_mask = torch.zeros_like(bool_masked_pos)
            with autocast_ctx:
                event_logits = model(
                    images,
                    full_mask,
                    physics_label=None,
                    physics_label_soft=None,
                    event_only_full_visible=True,
                )
                event_loss = compute_event_loss(
                    event_logits,
                    event_label,
                    event_score=event_score,
                    event_label_soft=event_label_soft,
                )
                event_acc = (
                    event_logits.argmax(dim=-1) == event_label
                ).float().mean()
                loss = event_loss_weight * event_loss

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(2)
            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') \
                and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss, optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(jepa_loss=0.0)
            metric_logger.update(sigreg_loss=0.0)
            metric_logger.update(balance_loss=0.0)
            metric_logger.update(physics_loss=0.0)
            metric_logger.update(general_loss=0.0)
            metric_logger.update(event_loss=event_loss.item())
            metric_logger.update(event_acc=event_acc.item())
            min_lr, max_lr = 10.0, 0.0
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            metric_logger.update(grad_norm=grad_norm)
            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(event_loss=event_loss.item(), head="loss")
                log_writer.update(event_acc=event_acc.item(), head="metric")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.set_step()
            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step)
            continue

        # ── 阶段一专用轻量路径 ───────────────────────────────────────────────
        #   只跑 encoder（到 block11 出 cls_scores），用 fp32 算 router_loss。
        #   完全绕开 predictor / target encoder / sigreg / autocast，
        #   既消除随机权重下 JEPA/sigreg 的 NaN，又省一半以上算力。
        if stage1:
            core = getattr(model, 'module', model)
            enc = core.encoder
            B_ = images.size(0)
            N_all = enc.patch_embed.num_patches
            # context path（只 visible tokens），fp32
            _ = enc(images, bool_masked_pos, physics_label, physics_label_soft)
            _pl = enc._physics_loss if enc._physics_loss is not None else images.new_zeros(())
            _gl = enc._general_loss if enc._general_loss is not None else images.new_zeros(())
            router_loss = 0.5 * _pl + 0.5 * _gl
            loss = router_loss
            jepa_loss = images.new_zeros(())
            sigreg_loss = images.new_zeros(())
            balance_loss = enc._balance_loss if enc._balance_loss is not None else images.new_zeros(())

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(2)

            optimizer.zero_grad()
            loss.backward()
            if max_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            else:
                grad_norm = utils.get_grad_norm_(model.parameters())
            optimizer.step()
            loss_scale_value = 0
            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(jepa_loss=jepa_loss.item())
            metric_logger.update(sigreg_loss=sigreg_loss.item())
            metric_logger.update(balance_loss=balance_loss.item())
            metric_logger.update(physics_loss=_pl.item())
            metric_logger.update(general_loss=_gl.item())
            min_lr, max_lr = 10., 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"]); max_lr = max(max_lr, group["lr"])
            metric_logger.update(lr=max_lr); metric_logger.update(min_lr=min_lr)
            wd_v = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0: wd_v = group["weight_decay"]
            metric_logger.update(weight_decay=wd_v)
            metric_logger.update(grad_norm=grad_norm)
            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(physics_loss=_pl.item(), head="loss")
                log_writer.update(general_loss=_gl.item(), head="loss")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()
            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step)
            continue  # 跳过下面的常规（端到端）路径

        if loss_scaler is None:
            model_output = model(
                images, bool_masked_pos, physics_label, physics_label_soft)
            if has_event_label:
                pred, target, event_logits = model_output
            else:
                pred, target = model_output
                event_logits = None
            (loss, jepa_loss, sigreg_loss, balance_loss, physics_loss,
             general_loss, event_loss, event_acc) = _calc_loss(
                 pred, target, event_logits)
        else:
            with autocast_ctx:
                model_output = model(
                    images, bool_masked_pos, physics_label, physics_label_soft)
                if has_event_label:
                    pred, target, event_logits = model_output
                else:
                    pred, target = model_output
                    event_logits = None
                (loss, jepa_loss, sigreg_loss, balance_loss, physics_loss,
                 general_loss, event_loss, event_acc) = _calc_loss(
                     pred, target, event_logits)

        loss_value = loss.item()

        # ── NaN 诊断（仅前2步打印）──────────────────────────────────
        if step < 2:
            _pl = getattr(encoder, '_physics_loss', None)
            _gl = getattr(encoder, '_general_loss', None)
            _pl_v = _pl.item() if _pl is not None else 0.0
            _gl_v = _gl.item() if _gl is not None else 0.0
            print(f"[DIAG step{step}] jepa={jepa_loss.item():.4f} "
                  f"sigreg={sigreg_loss.item():.4f} "
                  f"balance={balance_loss.item():.4f} "
                  f"physics={_pl_v:.4f} general={_gl_v:.4f} "
                  f"jepa_finite={torch.isfinite(jepa_loss).item()} "
                  f"sigreg_finite={torch.isfinite(sigreg_loss).item()} "
                  f"physics_finite={_pl is None or torch.isfinite(_pl).item()} "
                  f"general_finite={_gl is None or torch.isfinite(_gl).item()}")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        optimizer.zero_grad()
        if loss_scaler is None:
            loss.backward()
            if max_norm is None:
                grad_norm = utils.get_grad_norm_(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            optimizer.step()
            loss_scale_value = 0
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') \
                and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss, optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        # ── NaN 梯度诊断（仅前2步，找出哪些参数梯度nan）──────────────
        if step < 2:
            nan_params = []
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nan_params.append(name)
            if nan_params:
                print(f"[DIAG step{step}] {len(nan_params)} 个参数梯度NaN，前10个：")
                for nm in nan_params[:10]:
                    print(f"    {nm}")
            else:
                print(f"[DIAG step{step}] 所有参数梯度正常")

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(jepa_loss=jepa_loss.item())
        metric_logger.update(sigreg_loss=sigreg_loss.item())
        metric_logger.update(balance_loss=balance_loss.item())
        metric_logger.update(physics_loss=physics_loss.item())
        metric_logger.update(general_loss=general_loss.item())
        metric_logger.update(event_loss=event_loss.item())
        metric_logger.update(event_acc=event_acc.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value,              head="loss")
            log_writer.update(loss_scale=loss_scale_value,  head="opt")
            log_writer.update(lr=max_lr,                    head="opt")
            log_writer.update(min_lr=min_lr,                head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm,          head="opt")
            log_writer.update(jepa_loss=jepa_loss.item(),   head="loss")
            log_writer.update(sigreg_loss=sigreg_loss.item(), head="loss")
            log_writer.update(balance_loss=balance_loss.item(), head="loss")
            log_writer.update(physics_loss=physics_loss.item(), head="loss")
            log_writer.update(general_loss=general_loss.item(), head="loss")
            log_writer.update(event_loss=event_loss.item(), head="loss")
            log_writer.update(event_acc=event_acc.item(), head="metric")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
