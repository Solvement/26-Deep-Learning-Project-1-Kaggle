"""Distill an ensemble of scratch-trained Saber checkpoints into one WRN student.

Teachers are earlier Saber models trained from scratch on the official
training images only (rules section 5 permits such teachers). Teachers see the
exact augmented/mixed student batch ("consistent teaching"). The student keeps
the frozen train.py recipe; only the loss changes:
    (1-alpha) * mixed CE + alpha * T^2 * KL(mean teacher softmax_T || student_T)
Hidden test images are never loaded here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import seed_all, environment  # noqa: E402  (isolates cuDNN before torch)
from runtime import write_json as _write_json_once  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from data import load_training_arrays  # noqa: E402
from model import build_model, count_parameters  # noqa: E402
from research.resnet18 import PreActResNet18  # noqa: E402
from train import DEFAULT_CONFIG, augment, evaluate, normalize, schedule, code_provenance  # noqa: E402
from train import save_checkpoint as _save_once  # noqa: E402


def _retry(fn, *args):
    # OneDrive briefly locks synced files; retry the atomic rename instead of crashing mid-run.
    for _ in range(30):
        try:
            return fn(*args)
        except PermissionError:
            time.sleep(2)
    raise PermissionError(f"Could not replace {args[0]} after 30 retries")


def save_checkpoint(path, payload):
    return _retry(_save_once, path, payload)


def write_json(path, value):
    return _retry(_write_json_once, path, value)

TEACHER_SETS = {
    # 40k teachers: same recipes as the 50k set below, so the 40k run validates it on dev.
    "dev40k": ["results/research/wrn22_sam_s2026/best.pt",
               "results/research/wrn34_trivialaugment_bf16_s2026/best.pt",
               "results/research/wrn34_3_s2026/best.pt",
               "results/regularized/best.pt"],
    "full50k": ["results/research/sam_full_refit_nov04_train_20261104_v1/best.pt",
                "results/research/trivialaugment_full_refit_train_20261104_v1/best.pt",
                "results/research/wrn34_3_full50k_s2026/best.pt",
                "results/final_full/best.pt"],
}
EXTRA_40K = ["results/research/wrn22_resizemix300_bf16_s2026/best.pt",
             "results/research/colab_wrn34_3_300_fp16_s2027_v3/candidate/best.pt",
             "results/research/wrn22_sam_s2027/best.pt",
             "results/research/wrn34_trivialaugment_bf16_s2027/best.pt"]
TEACHER_SETS["dev40k_8"] = TEACHER_SETS["dev40k"] + EXTRA_40K
TEACHER_SETS["full50k_8"] = TEACHER_SETS["full50k"] + EXTRA_40K  # 40k-trained teachers are still official-data scratch models
R18_40K = ["results/research/r18w64_dev40k_s2026_v1/final_model.pt"]  # 4-stage ResNet-18 teacher for diversity
TEACHER_SETS["dev40k_r18"] = TEACHER_SETS["dev40k"] + R18_40K
TEACHER_SETS["full50k_r18"] = TEACHER_SETS["full50k_8"] + R18_40K
TEACHER_SETS["dev40k_none"] = []  # plain CE training (alpha forced to 0), e.g. a new diverse teacher
TEACHER_SETS["full50k_none"] = []
STUDENTS = {"wrn22_4": None, "resnet18_40": {"arch": "preact_resnet18", "base": 40, "num_classes": 10},
            "resnet18_64": {"arch": "preact_resnet18", "base": 64, "num_classes": 10}}
PARAM_LIMIT = 5_000_000


def make_model(cfg):
    return PreActResNet18(cfg["base"], cfg["num_classes"]) if cfg.get("arch") == "preact_resnet18" else build_model(cfg)


def load_teachers(paths, device):
    teachers = []
    for path in paths:
        ckpt = torch.load(ROOT/path, map_location="cpu", weights_only=False)
        model = make_model(ckpt["model_config"])
        model.load_state_dict(ckpt["model"])
        teachers.append(model.to(device=device, memory_format=torch.channels_last).eval().requires_grad_(False))
    return teachers


@torch.no_grad()
def teacher_probs(teachers, x, temperature):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.stack([F.softmax(t(x).float()/temperature, 1) for t in teachers]).mean(0)


@torch.no_grad()
def teacher_dev_accuracy(teachers, images, labels):
    correct = 0
    for start in range(0, len(labels), 500):
        x = normalize(images[start:start+500])
        p = teacher_probs(teachers, x, 1.0) + teacher_probs(teachers, x.flip(-1), 1.0)
        correct += p.argmax(1).eq(labels[start:start+500]).sum().item()
    return correct/len(labels)


def train(args):
    config = dict(DEFAULT_CONFIG, epochs=args.epochs, train_all=args.teachers.startswith("full50k"),
                  kd_alpha=args.alpha if TEACHER_SETS[args.teachers] else 0.0, student=args.student, kd_temperature=args.temperature, teachers=TEACHER_SETS[args.teachers])
    seed_all(config["seed"])
    device = torch.device("cuda")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    images, labels, train_ids, val_ids = load_training_arrays(ROOT/"data")
    if config["train_all"]:
        train_ids, val_ids = np.arange(len(labels)), np.array([], dtype=np.int64)
    train_x = torch.as_tensor(images[train_ids], device=device); train_y = torch.as_tensor(labels[train_ids], device=device)
    val_x = torch.as_tensor(images[val_ids], device=device); val_y = torch.as_tensor(labels[val_ids], device=device)

    teachers = load_teachers(config["teachers"], device)
    model_config = STUDENTS[args.student] or {k: config[k] for k in ("depth", "widen_factor", "dropout", "num_classes")}
    model = make_model(model_config).to(device=device, memory_format=torch.channels_last)
    params = count_parameters(model, trainable_only=True)
    if (params > PARAM_LIMIT or count_parameters(model) > PARAM_LIMIT) and not args.teacher_only:
        raise ValueError(f"Student exceeds 5M parameters: {params}")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                          weight_decay=config["weight_decay"], nesterov=True)
    steps_per_epoch = math.ceil(len(train_ids)/config["batch_size"]); total_steps = config["epochs"]*steps_per_epoch
    start_epoch, updates, history = 1, 0, []
    if args.resume:
        last = torch.load(out/"last.pt", map_location=device, weights_only=False)
        if last["config"] != config: raise ValueError("Resume config mismatch")
        model.load_state_dict(last["model"]); ema.load_state_dict(last["ema"]); opt.load_state_dict(last["optimizer"])
        start_epoch, updates, history = last["epoch"]+1, last["updates"], last["history"]
        random.setstate(last["python_rng"]); np.random.set_state(last["numpy_rng"])
        torch.set_rng_state(last["torch_rng"].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in last["cuda_rng"]])
    elif (out/"last.pt").exists():
        raise FileExistsError("Existing run: pass --resume or a new output dir")
    else:
        write_json(out/"config.json", config); write_json(out/"environment.json", environment())
        teacher_acc = None if config["train_all"] or not teachers else teacher_dev_accuracy(teachers, val_x, val_y)
        write_json(out/"provenance.json", {"code": code_provenance(), "student_parameters": params,
            "train_examples": len(train_ids), "validation_examples": len(val_ids), "teacher_dev_flip_accuracy": teacher_acc})
        print(json.dumps({"event": "start", "params": params, "train": len(train_ids), "teacher_dev_acc": teacher_acc}), flush=True)

    model_params, ema_params = list(model.parameters()), list(ema.parameters())
    T, alpha = config["kd_temperature"], config["kd_alpha"]
    started = time.perf_counter()
    for epoch in range(start_epoch, config["epochs"]+1):
        t0 = time.perf_counter(); model.train()
        order = torch.randperm(len(train_y), device=device)
        loss_sum = torch.zeros((), device=device)
        for batch, start in enumerate(range(0, len(order), config["batch_size"])):
            lr = schedule((epoch-1)*steps_per_epoch+batch, total_steps, steps_per_epoch, config)
            for g in opt.param_groups: g["lr"] = lr
            ids = order[start:start+config["batch_size"]]; y = train_y[ids]
            x = augment(train_x[ids], config["cutout"])
            mix = random.random() < config["mixup_probability"]
            lam = float(np.random.beta(config["mixup_alpha"], config["mixup_alpha"])) if mix else 1.0
            perm = torch.randperm(len(y), device=device) if mix else None
            if mix: x = x*lam + x[perm]*(1-lam)
            soft = teacher_probs(teachers, x, T) if alpha > 0 else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            hard = F.cross_entropy(logits, y)*lam
            if mix: hard = hard + F.cross_entropy(logits, y[perm])*(1-lam)
            loss = hard if soft is None else (1-alpha)*hard + alpha*F.kl_div(F.log_softmax(logits/T, 1), soft, reduction="batchmean")*T*T
            if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss epoch {epoch} batch {batch}")
            loss.backward(); opt.step(); updates += 1
            with torch.no_grad():
                torch._foreach_lerp_(ema_params, model_params, 1-min(config["ema_decay"], (1+updates)/(10+updates)))
                loss_sum += loss.detach()
        with torch.no_grad():
            for a, b in zip(ema.buffers(), model.buffers()): a.copy_(b)
        rec = {"epoch": epoch, "loss": loss_sum.item()/steps_per_epoch, "lr": lr, "epoch_seconds": time.perf_counter()-t0,
               "elapsed": time.perf_counter()-started}
        if len(val_y):
            rec["dev_model"] = evaluate(model, val_x, val_y, device)["accuracy"]
            rec["dev_ema"] = evaluate(ema, val_x, val_y, device)["accuracy"]
            rec["dev_model_flip"] = evaluate(model, val_x, val_y, device, flip_tta=True)["accuracy"]
        history.append(rec)
        meta = {"model_config": model_config, "config": config, "epoch": epoch, "parameter_count": params}
        if epoch == config["epochs"]:  # fixed final epoch, both variants kept; no dev-based epoch selection
            save_checkpoint(out/"final_model.pt", {**meta, "model": model.state_dict(), "variant": "model"})
            save_checkpoint(out/"final_ema.pt", {**meta, "model": ema.state_dict(), "variant": "ema"})
        save_checkpoint(out/"last.pt", {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": opt.state_dict(),
            "epoch": epoch, "updates": updates, "config": config, "history": history,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()})
        write_json(out/"history.json", history)
        print(json.dumps(rec), flush=True)
    write_json(out/"metrics.json", {"status": "complete", "epochs": config["epochs"], "updates": updates, "final": history[-1]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teachers", choices=sorted(TEACHER_SETS), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--student", choices=sorted(STUDENTS), default="wrn22_4")
    p.add_argument("--teacher-only", action="store_true", help="allow >5M params: model is only a teacher, never submitted")
    train(p.parse_args())


if __name__ == "__main__":
    main()
