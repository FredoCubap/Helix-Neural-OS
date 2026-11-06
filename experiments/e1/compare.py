"""E1: misma pregunta con la base sola, con el LoRA de Python y con el de SQL.

Requiere llama-server corriendo en 127.0.0.1:8080 con los LoRA cargados en este orden:
  --lora models/lora/python.gguf,models/lora/sql.gguf --lora-init-without-apply
Cada petición elige los adaptadores con el campo `lora` (id + scale).
OJO: `"lora": []` NO apaga los adaptadores, aplica la escala global del servidor (1.0 para ambos, aun con
--lora-init-without-apply). Por eso siempre se mandan las escalas de TODOS los adaptadores.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"


def scales(py=0.0, sql=0.0):
    return [{"id": 0, "scale": py}, {"id": 1, "scale": sql}]


CONFIGS = {
    "base": scales(),
    "python": scales(py=1.0),
    "sql": scales(sql=1.0),
}
PROMPTS = {
    "python": "Write a Python function that removes duplicates from a list while keeping order.",
    # Formato con el que se entrenó el LoRA de SQL (ver su ficha, F-30).
    "sql": "Convert the following natural language question to SQL:\n\n"
           "Database: concert_singer\nQuestion: How many singers do we have?\n\nSQL:",
}


def ask(prompt, lora):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "lora": lora,
        "temperature": 0,
        "max_tokens": 120,
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    dt = time.perf_counter() - t0
    return data["choices"][0]["message"]["content"], dt, data.get("timings", {})


if __name__ == "__main__":
    for pname, prompt in PROMPTS.items():
        print(f"\n######## Pregunta: {pname}")
        for cname, lora in CONFIGS.items():
            text, dt, tm = ask(prompt, lora)
            print(f"\n--- [{cname}] {dt:.2f} s total | prompt {tm.get('prompt_ms', 0):.0f} ms"
                  f" | gen {tm.get('predicted_per_second', 0):.1f} t/s")
            print(text.strip())
