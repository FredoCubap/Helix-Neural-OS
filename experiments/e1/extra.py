"""E1, pruebas extra contra un llama-server ya levantado (ver README).

  python extra.py mix    -> mezcla de adaptadores con escalas parciales
  python extra.py long   -> costo de lectura del prompt según largo, base vs. LoRA
  python extra.py greedy -> respuestas greedy de la base y del LoRA SQL (para comparar Q8_0 vs. f16 vs. PEFT)
"""
import json
import random
import sys
import urllib.request

from compare import scales  # siempre escalas explícitas: "lora": [] aplica ambos a 1.0

BASE_URL = "http://127.0.0.1:8080"
SQL_RAW = ("Convert the following natural language question to SQL:\n\n"
           "Database: concert_singer\nQuestion: How many singers do we have?\n\nSQL:")
PY_CHAT = "Write a Python function that removes duplicates from a list while keeping order."


def post(path, body):
    req = urllib.request.Request(BASE_URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def completion(prompt, lora, n=60, cache=True):
    return post("/completion", {"prompt": prompt, "lora": lora, "temperature": 0,
                                "n_predict": n, "cache_prompt": cache})


def chat(prompt, lora, n=120):
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}],
                                      "lora": lora, "temperature": 0, "max_tokens": n})
    return d["choices"][0]["message"]["content"]


def mix():
    configs = {
        "base": scales(),
        "py 1.0": scales(py=1.0),
        "sql 1.0": scales(sql=1.0),
        "py 0.5 + sql 0.5": scales(0.5, 0.5),
        "py 1.0 + sql 1.0": scales(1.0, 1.0),
        "py 0.5": scales(py=0.5),
    }
    for name, lora in configs.items():
        sql = completion(SQL_RAW, lora)["content"].strip().splitlines()[0][:110]
        py = chat(PY_CHAT, lora)
        has_code = "def " in py
        print(f"[{name:17}] SQL: {sql!r}\n{'':19}Python: {'con código' if has_code else 'SIN código'} | {py[:90]!r}")


def long():
    random.seed(0)
    words = "the model reads this filler text about singers concerts tables and python lists".split()
    print("tokens_prompt | config | prompt_ms | t/s prompt")
    for n_words in (50, 400, 1600):
        for name, lora in (("base", scales()), ("py", scales(py=1.0)), ("ambos", scales(1.0, 1.0))):
            # Texto distinto en cada petición y sin caché, para que se lea el prompt completo.
            text = " ".join(random.choice(words) for _ in range(n_words))
            t = completion(text + "\nSummary:", lora, n=1, cache=False)["timings"]
            print(f"{t['prompt_n']:>13} | {name:6} | {t['prompt_ms']:>9.0f} | {t['prompt_per_second']:>9.0f}")


def greedy():
    for name, lora in (("base", scales()), ("sql", scales(sql=1.0))):
        print(f"[{name}] {completion(SQL_RAW, lora, n=40)['content'][:160]!r}")


if __name__ == "__main__":
    {"mix": mix, "long": long, "greedy": greedy}[sys.argv[1]]()
