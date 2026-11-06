# Helix Neural OS

Proyecto de investigación: un **sistema operativo neural modular**. Un núcleo pequeño que razona (inteligencia
fluida) y módulos de conocimiento intercambiables en caliente, hoy adaptadores LoRA (inteligencia cristalizada),
que se eligen, combinan y retiran por petición.

Punto de partida: `docs/guia-arquitectura-neural-os.md` (documento original, **con errores conocidos**; ver
`docs/mapa-investigacion.md` §2 para qué es real y qué no).

Destino concreto: NPC de **Kenshi** con lore verdadero + voz de facción, servidos a través de un mod tipo
Sentient Sands [F-33], que acepta cualquier API compatible con OpenAI.

## Estado (2026-11-06)

| Experimento | Estado | Resultado en una línea | Dónde |
|---|---|---|---|
| E0 Hardware | ✅ | Laboratorio en CPU + iGPU (Vulkan); entrenamiento en Colab T4 | `docs/hardware.md` |
| E1 Cambio de LoRA por petición | ✅ | Funciona en local; cambiar es gratis, cada LoRA activo cuesta ~20 % más de lectura de prompt; el nombre del adaptador no garantiza su especialidad | `experiments/e1/` |
| E2 ¿Un LoRA aprende el lore de Kenshi? | 🔄 en curso, rama `exp/e2-kenshi` | Aprende la forma, recombina entidades (~26 % en paráfrasis, cota superior), aumenta la alucinación; separar lore de voces mantiene las respuestas neutrales | `experiments/e2/README.md` (en la rama) |

Próximos pasos de E2: barrido de épocas/lr, aumento de conocimiento con paráfrasis, RAG vs. LoRA (E3), bases más
grandes. Plan general en `docs/mapa-investigacion.md`.

## Estructura

```
CLAUDE.md                 reglas del proyecto (léelas primero)
docs/
  guia-arquitectura-...   guía original (punto de partida)
  mapa-investigacion.md   qué es real, literatura por área, preguntas, experimentos
  fuentes.md              registro de TODA fuente consultada ([F-NN], [S-NN], datos propios [D-NN])
  hardware.md             equipo, entorno, mediciones
experiments/eN/           un directorio por experimento: README con resultados + scripts + corridas
requirements/cpu.txt      dependencias fijadas (perfil CPU)
.venv/ .tools/ models/ data/   fuera de git (entorno, llama.cpp, modelos, datos privados)
```

## Puesta en marcha (Windows, perfil CPU)

1. Python + entorno del proyecto con [uv](https://docs.astral.sh/uv/) (nunca instalar en el Python global):
   ```
   uv venv .venv --python 3.14
   uv pip install -p .venv --index-strategy unsafe-best-match --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/cpu.txt
   ```
   (`--index-strategy unsafe-best-match` hace falta porque `torch==2.15.0+cpu` solo está en el índice de PyTorch y el
   resto solo en PyPI; ambos índices son oficiales. Verificado con `--dry-run` en un entorno nuevo el 2026-11-06.)
2. llama.cpp **build b12121** (binario y scripts de conversión de la misma build):
   ```
   winget install ggml.llamacpp          # binarios Vulkan
   git clone --depth 1 --branch b12121 https://github.com/ggml-org/llama.cpp .tools/llama.cpp
   uv pip install -p .venv -e .tools/llama.cpp/gguf-py
   ```
3. Modelo base (revisión fijada) y su GGUF propio:
   ```
   .venv/Scripts/hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir models/Qwen2.5-0.5B-Instruct
   .venv/Scripts/python .tools/llama.cpp/convert_hf_to_gguf.py models/Qwen2.5-0.5B-Instruct --outtype f16 --outfile models/qwen2.5-0.5b-instruct-f16.gguf
   ```
4. Cada experimento explica en su README cómo reproducirlo (servidor, LoRA, scripts).

Entrenamiento con GPU: `experiments/e2/colab_train.ipynb` (en la rama de E2) corre los mismos scripts en una T4.

## Datos privados

El dataset de Kenshi (D-01) es de Fredo y deriva de textos de Lo-Fi Games: **no se sube a git ni se publica**.
Se pide a Fredo y se coloca en `data/kenshi/Kenshi_Dataset-ES_final.jsonl` (sha256 empieza por `f40b00d6e98c0114`).

## Cómo trabajamos (resumen de `CLAUDE.md`)

- **Español** para todo lo que escribimos (docs, comentarios, commits); el código y sus identificadores, en inglés.
- **Toda fuente consultada se registra** en `docs/fuentes.md` con tipo, nivel de lectura y fecha; se cita `[F-NN]`.
- **`main` = investigación**; ramas cortas `exp/...` y `feat/...` que se fusionan al cerrar; el hardware es configuración, no rama.
- **Fabricar vs. usar**: entrenar puede ir a GPU (Colab, PCs NVIDIA) con el mismo script; servir es local; se guardan solo adaptadores.
- **Medir con controles**: cada resultado se verifica contra una referencia (p. ej. PEFT vs. GGUF, control con escalas en 0,
  control de eco en la evaluación). Dos trampas ya pisadas: `"lora": []` en llama-server aplica todos los adaptadores, y
  puntuar por claves sin descartar las que ya están en la pregunta.
