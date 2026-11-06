"""E2: separa el dataset de Kenshi (D-01) en bloques y marca filas defectuosas.

El JSONL viene ordenado por bloques (revisado a mano el 2026-11-06):
  0–208     lore_narrador  preguntas de lore respondidas en voz enciclopédica
  209–237   lore_voz       las mismas preguntas de lore respondidas por un NPC (con acotaciones)
  238–890   escenas        escenas de rol con NPC (tenderos, matones, esqueletos...); acotaciones sin marcar
  891–1028  facciones      voz de facción (Shek, Nación Sagrada) con acotaciones entre paréntesis

Uso: python prepare.py   -> escribe data/kenshi/prepared/<bloque>.jsonl y imprime el resumen.
Los datos no se suben a git (texto derivado de Lo-Fi Games).
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/kenshi/Kenshi_Dataset-ES_final.jsonl"
OUT = ROOT / "data/kenshi/prepared"
BLOCKS = [("lore_narrador", 0, 208), ("lore_voz", 209, 237), ("escenas", 238, 890), ("facciones", 891, 1028)]


def block_of(i):
    for name, lo, hi in BLOCKS:
        if lo <= i <= hi:
            return name
    raise ValueError(f"fila {i} fuera de los bloques conocidos")


def main():
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == BLOCKS[-1][2] + 1, f"se esperaban {BLOCKS[-1][2] + 1} filas, hay {len(rows)}"
    OUT.mkdir(parents=True, exist_ok=True)
    kept, dropped = collections.defaultdict(list), collections.Counter()
    for i, r in enumerate(rows):
        conv = r["conversations"]
        assert [m["from"] for m in conv] == ["human", "gpt"], f"fila {i}: roles inesperados"
        q, a = conv[0]["value"].strip(), conv[1]["value"].strip()
        b = block_of(i)
        if q == a:  # defecto: la respuesta es copia de la pregunta
            dropped[b] += 1
            continue
        kept[b].append({"id": i, "block": b, "question": q, "answer": a})
    for b, items in kept.items():
        with open(OUT / f"{b}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
    for name, lo, hi in BLOCKS:
        print(f"{name:14} filas {lo:4}–{hi:4}: {len(kept[name]):4} útiles, {dropped[name]:3} descartadas (respuesta = pregunta)")


if __name__ == "__main__":
    main()
