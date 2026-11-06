"""E2: evalúa un modelo servido por llama-server contra el banco `eval_bank.jsonl`.

Categorías:
  parafrasis   hecho entrenado, pregunta reescrita        -> correcto si aparecen todas las claves `must`
  literal      la pregunta exacta del dataset (src[0])     -> idem (mide memorización frente a paráfrasis)
  holdout      hecho apartado de todo el entrenamiento     -> idem (debería quedar ~0: control de fuga)
  inexistente  entidad inventada                           -> correcto si se abstiene (heurística; revisar a mano)
  general      conocimiento fuera de Kenshi                -> idem `must` (olvido catastrófico)
  perspectiva  pregunta neutral                            -> `must` presente y ninguna clave `must_not` (voz de facción)

Las claves se comparan en minúsculas y sin tildes. Cada grupo de `must` es una lista de alternativas.
Control de eco: una alternativa que ya aparece en la pregunta no cuenta (el modelo podría repetirla sin saber nada).
Si un grupo no tiene ninguna alternativa ausente de la pregunta, el ítem es "no puntuable" y queda fuera del total.
"inexistente" mide "no inventa": cuenta como correcto rechazar o decir que no existe.

Uso:
  python eval.py --selftest                       # valida las claves contra las respuestas del dataset
  python eval.py TAG [--lora 0:1.0,1:0.0]         # evalúa; guarda respuestas en data/kenshi/runs/eval_TAG.jsonl
  python eval.py --rescore TAG                    # vuelve a puntuar respuestas ya guardadas
"""
import collections
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URL = "http://127.0.0.1:8080/v1/chat/completions"
ABSTAIN = ["no existe", "no hay informacion", "no tengo informacion", "no conozco", "desconozco", "no se sabe",
           "no estoy seguro", "no hay registro", "no aparece", "no tengo datos", "no puedo proporcionar"]


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def hit(text, groups, question=""):
    t, q = norm(text), norm(question)
    return all(any(norm(alt) in t and norm(alt) not in q for alt in g) for g in groups)


def scorable(item):
    q = norm(item["q"])
    return all(any(norm(alt) not in q for alt in g) for g in item["must"])


def any_hit(text, groups):
    t = norm(text)
    return any(norm(alt) in t for g in groups for alt in g)


def score(item, answer):
    """True/False, o None si el ítem no es puntuable sin eco."""
    if item["cat"] == "inexistente":
        return any(a in norm(answer) for a in ABSTAIN)
    if not scorable(item):
        return None
    ok = hit(answer, item["must"], item["q"])
    if item.get("must_not"):
        ok = ok and not any_hit(answer, item["must_not"])
    return ok


def load():
    bank = [json.loads(x) for x in (HERE / "eval_bank.jsonl").read_text(encoding="utf-8").splitlines()]
    lore = {json.loads(x)["id"]: json.loads(x) for x in
            (ROOT / "data/kenshi/prepared/lore_narrador.jsonl").read_text(encoding="utf-8").splitlines()}
    literal = [dict(it, id="lit" + it["id"][3:], cat="literal", q=lore[it["src"][0]]["question"])
               for it in bank if it["cat"] == "parafrasis"]
    return bank + literal, lore


def ask(q, lora):
    body = {"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 120}
    if lora is not None:
        body["lora"] = lora
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def selftest():
    items, lore = load()
    bad = 0
    for it in items:
        if not it["src"] or it["cat"] == "literal":
            continue
        for s in it["src"]:
            if not hit(lore[s]["question"] + " " + lore[s]["answer"], it["must"]):  # se entrena con el par
                bad += 1
                print(f"CLAVE NO ENCONTRADA {it['id']} en fila {s}: {it['must']}")
    print("selftest:", "OK" if bad == 0 else f"{bad} fallos")


def parse_lora(spec):
    # "0:1.0,1:0.0" -> [{"id":0,"scale":1.0},{"id":1,"scale":0.0}]; siempre todas las escalas (ver E1).
    return [{"id": int(i), "scale": float(s)} for i, s in (p.split(":") for p in spec.split(","))]


def report(tag, results):
    tally = collections.defaultdict(lambda: [0, 0, 0])  # aciertos, puntuables, no puntuables
    for r in results:
        t = tally[r["cat"]]
        if r["ok"] is None:
            t[2] += 1
        else:
            t[0] += r["ok"]
            t[1] += 1
    print(f"== {tag}")
    for cat in ("literal", "parafrasis", "holdout", "perspectiva", "inexistente", "general"):
        ok, n, skip = tally[cat]
        extra = f"  ({skip} no puntuables por eco)" if skip else ""
        print(f"  {cat:12} {ok:3}/{n:<3} {100 * ok / max(n, 1):5.1f} %{extra}")


def main(tag, lora):
    items, _ = load()
    out = ROOT / f"data/kenshi/runs/eval_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            a = ask(it["q"], lora)
            r = {"id": it["id"], "cat": it["cat"], "q": it["q"], "ok": score(it, a), "answer": a}
            results.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(tag, results)


def rescore(tag):
    items = {it["id"]: it for it in load()[0]}
    path = ROOT / f"data/kenshi/runs/eval_{tag}.jsonl"
    results = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for r in results:
        r["ok"] = score(items[r["id"]], r["answer"])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    report(tag, results)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--rescore" in sys.argv:
        rescore(sys.argv[sys.argv.index("--rescore") + 1])
    else:
        lora = parse_lora(sys.argv[sys.argv.index("--lora") + 1]) if "--lora" in sys.argv else None
        main(sys.argv[1], lora)
