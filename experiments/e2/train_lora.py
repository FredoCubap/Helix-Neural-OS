"""E2: entrena un LoRA sobre Qwen2.5-0.5B-Instruct en CPU (fp32) con PEFT y un bucle mínimo de PyTorch.

Pérdida solo sobre los tokens de la respuesta (assistant). Misma forma de LoRA que los de E1 (r=16, alpha=32,
los 7 módulos) para que el tamaño y el costo sean comparables.

Uso: python train_lora.py VARIANTE [--epochs N] [--lr X] [--bs B] [--limit K]
  VARIANTE = lore | voces | todo  (data/kenshi/train/<variante>.jsonl, ver build_train.py)
  --limit K entrena solo con K ejemplos (para medir velocidad)
Salida: models/lora/e2-<variante>/ (adapter_model.safetensors + train_log.json)
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/Qwen2.5-0.5B-Instruct"


def encode(tok, messages):
    full = tok.apply_chat_template(messages, tokenize=False)
    prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False).input_ids
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    assert ids[:n_prompt] == tok(prompt, add_special_tokens=False).input_ids, "el prompt no es prefijo del ejemplo"
    labels = [-100] * n_prompt + ids[n_prompt:]
    return ids, labels


def batches(examples, bs, pad_id, rng):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [examples[j] for j in order[i:i + bs]]
        n = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), n), pad_id)
        lab = torch.full((len(chunk), n), -100)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, :len(x)], lab[k, :len(y)], att[k, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    rows = [json.loads(x) for x in (ROOT / f"data/kenshi/train/{args.variant}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    examples = [encode(tok, r["messages"]) for r in rows]
    n_tok = sum(len(x) for x, _ in examples)
    n_ans = sum(sum(1 for t in y if t != -100) for _, y in examples)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()

    steps = args.epochs * math.ceil(len(examples) / args.bs)
    warm = max(1, steps // 20)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    print(f"[{args.variant}] {len(examples)} ejemplos, {n_tok} tokens ({n_ans} de respuesta), {steps} pasos")
    log, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        ep_loss, ep_n = 0.0, 0
        for ids, lab, att in batches(examples, args.bs, tok.pad_token_id, rng):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            ep_loss += loss.item(); ep_n += 1
        el = time.time() - t0
        rec = {"epoch": ep + 1, "loss": ep_loss / ep_n, "elapsed_s": round(el, 1)}
        log.append(rec)
        print(f"  época {ep + 1}/{args.epochs}  loss {rec['loss']:.4f}  {el / 60:.1f} min  "
              f"({n_tok * (ep + 1) / el:.0f} tok/s)", flush=True)

    out = ROOT / f"models/lora/e2-{args.variant}{'-limit' + str(args.limit) if args.limit else ''}"
    model.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "examples": len(examples), "tokens": n_tok,
                                                    "answer_tokens": n_ans, "log": log}, indent=2), encoding="utf-8")
    print("guardado en", out)


if __name__ == "__main__":
    main()
