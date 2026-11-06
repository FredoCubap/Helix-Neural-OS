"""E2: arma los conjuntos de entrenamiento a partir de los bloques de prepare.py.

Variantes:
  lore   lore_narrador
  voces  lore_voz + escenas + facciones
  todo   lore + voces
A todas se les quita cualquier fila (pregunta o respuesta) que coincida con un `anchor` de los ítems holdout de
eval_bank.jsonl, para que esos hechos no se filtren al entrenamiento.

Salida: data/kenshi/train/<variante>.jsonl con {"id", "block", "messages": [user, assistant]}.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT / "data/kenshi/prepared"
OUT = ROOT / "data/kenshi/train"
VARIANTS = {
    "lore": ["lore_narrador"],
    "voces": ["lore_voz", "escenas", "facciones"],
    "todo": ["lore_narrador", "lore_voz", "escenas", "facciones"],
}


def main():
    bank = [json.loads(x) for x in (HERE / "eval_bank.jsonl").read_text(encoding="utf-8").splitlines()]
    anchors = [re.compile(it["anchor"]) for it in bank if it.get("anchor")]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, blocks in VARIANTS.items():
        kept, removed = [], []
        for b in blocks:
            for line in (PREP / f"{b}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                text = (r["question"] + " " + r["answer"]).lower()
                if any(a.search(text) for a in anchors):
                    removed.append(r["id"])
                    continue
                kept.append({"id": r["id"], "block": b, "messages": [
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["answer"]}]})
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in kept)
        print(f"{name:6} {len(kept):4} ejemplos | quitados por holdout: {sorted(removed)}")


if __name__ == "__main__":
    main()
