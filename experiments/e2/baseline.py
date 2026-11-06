"""E2, línea base: ¿cuánto lore de Kenshi sabe el modelo base sin LoRA?

Toma N preguntas al azar (semilla fija) de lore_narrador, las pregunta a llama-server (base sola, sin --lora,
temperatura 0) y guarda pregunta, respuesta de referencia del dataset y respuesta del modelo para revisarlas a mano.
Uso: python baseline.py [N] [--ctx] > salida   (--ctx agrega un system prompt que dice que se habla de Kenshi)
"""
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8080/v1/chat/completions"


SYSTEM_CTX = "Responde sobre el videojuego Kenshi (Lo-Fi Games), su mundo y su lore."


def ask(q, ctx=False):
    msgs = ([{"role": "system", "content": SYSTEM_CTX}] if ctx else []) + [{"role": "user", "content": q}]
    body = {"messages": msgs, "temperature": 0, "max_tokens": 150}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 20
    ctx = "--ctx" in sys.argv
    rows = [json.loads(x) for x in (ROOT / "data/kenshi/prepared/lore_narrador.jsonl").read_text(encoding="utf-8").splitlines()]
    random.seed(42)
    for r in random.sample(rows, n):
        print(f"### [{r['id']}] {r['question']}\nREF: {r['answer']}\nBASE: {ask(r['question'], ctx)}\n")
