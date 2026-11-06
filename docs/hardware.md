# Hardware disponible y consecuencias

## Equipo local (medido el 2026-11-05)

| Componente | Valor |
|---|---|
| CPU | Intel Core i5-1235U (12.ª gen, 10 núcleos / 12 hilos, portátil de bajo consumo) |
| GPU | Intel Iris Xe integrada (~2 GB compartidos). **Sin GPU dedicada, sin CUDA.** |
| RAM | 39,7 GB |
| Disco libre (C:) | 343 GB |
| SO | Windows 11 Pro |
| Ollama | 0.40.14 (actualizado con winget el 2026-11-05), sin modelos descargados |
| llama.cpp | build b12121 (winget `ggml.llamacpp`, binarios **Vulkan** x64). `llama-cli --list-devices` detecta `Vulkan0: Intel(R) Iris(R) Xe Graphics (20301 MiB)` — memoria compartida con la RAM |

## Entorno de software (2026-11-06)

- `.venv` con uv 0.12.10 y Python 3.14.7: torch 2.15.0+cpu, transformers 5.22.1, peft 0.21.2 (lista completa en `requirements/cpu.txt`).
- Scripts de conversión: `.tools/llama.cpp` clonado en el tag `b12121` (commit `e56f006ed`), igual que el binario.
- Modelo base: `Qwen/Qwen2.5-0.5B-Instruct`, revisión `7ae557604adf67be50417f59c2c2f167def9a775` [F-27], en `models/Qwen2.5-0.5B-Instruct`. GGUF propio: `models/qwen2.5-0.5b-instruct-q8_0.gguf` (525 MB, generado con `convert_hf_to_gguf.py --outtype q8_0`).

## Medición base: Qwen2.5-0.5B Q8_0 (llama-bench, 2026-11-06)

| Backend | Capas en GPU (`-ngl`) | Prompt (pp256) | Generación (tg64) |
|---|---|---|---|
| CPU | 0 | 174 ± 36 t/s | 6,3 ± 1,4 t/s |
| Iris Xe (Vulkan) | 99 | 331 ± 28 t/s | **17,1 ± 0,2 t/s** |

→ **Conviene la Iris Xe por Vulkan** (~2,7× en generación). La CPU varía mucho entre corridas (portátil, ahorro de energía). Prueba de humo: "¿Cuál es la capital de Francia?" → "La capital de Francia es París."

## Entrenamiento: CPU local vs. Colab T4 (2026-11-06, `experiments/e2/train_lora.py`)

| Entorno | Velocidad (Qwen2.5-0.5B, LoRA r=16, fp32 / fp16 autocast) |
|---|---|
| CPU i5-1235U, 10 hilos (6 hilos da lo mismo; 2–4 hilos, peor) | ~46 tok/s |
| Colab Tesla T4 (torch 2.11.0+cu130) | ~2.450–3.070 tok/s |

→ Entrenar va a la T4 (regla "fabricar vs. usar" en `CLAUDE.md`); la CPU queda para pruebas con `--limit`.
En Colab hay que desinstalar `torchao` (PEFT 0.21.2 aborta con el 0.10.0 preinstalado).

## Qué no corre aquí

- **vLLM**: necesita GPU (CUDA/ROCm). Su backend de CPU es para Linux, no es nuestro caso. → toda la parte de servido de la guía queda fuera del equipo local.
- **QLoRA con bitsandbytes**: la cuantización 4-bit de bitsandbytes necesita CUDA. **Unsloth** también.
- **Entrenar LoRA sobre modelos de 7–8B**: imposible en tiempo razonable en CPU.

## Qué sí corre aquí

- **Aceleración por Vulkan**: la build de llama.cpp usa la Iris Xe con ~20 GB de memoria compartida; medido arriba, gana a la CPU.

- **Inferencia de modelos cuantizados GGUF** con Ollama o llama.cpp: modelos 1–4B con fluidez, 7–8B lento pero usable (40 GB de RAM alcanzan).
- **LoRA por petición con llama-server (llama.cpp)** [F-22]: `--lora` carga varios adaptadores, `--lora-init-without-apply` los carga sin aplicar, y cada petición elige cuáles usar y con qué peso (campo `lora`, con `id` y `scale`). Esto es un **sustituto local del multi-LoRA de vLLM**, y además permite mezclar adaptadores con pesos (como en LORAUTER).
- **Ollama**: tiene que verificarse si la versión instalada sigue soportando `ADAPTER` en el Modelfile; la doc actual no lo muestra [F-23]. Aun cuando lo soporte, sería un adaptador fijo por modelo, no un cambio por petición. → se usa para pruebas rápidas del modelo base, no como motor del sistema.
- **Entrenar LoRA chicos en CPU** con PEFT + Transformers (fp32/bf16) sobre modelos de 0,5–1,5B (p. ej. Qwen2.5‑0.5B/1.5B, SmolLM2) y corpus pequeños: lento, pero alcanza para los experimentos de capacidad y de Deep Context Distillation a escala reducida (E2, E3).
- **Edición de conocimiento (EasyEdit, AlphaEdit, μKE)** sobre modelos chicos tipo GPT‑2 / 0,5B: plausible en CPU, por confirmar.

## Estrategia

1. **Local = laboratorio a escala chica.** Las preguntas de fondo (¿un LoRA guarda hechos? ¿cuánto? ¿se puede editar?) se pueden responder con modelos de 0,5–1,5B. Si no funciona en chico, no hay que gastar GPU en grande.
2. **GPU en la nube solo cuando haga falta escalar** (Colab/Kaggle gratis con T4 de 16 GB, o alquiler por hora). Antes de usarla, decidir proveedor y presupuesto con Fredo.
3. Diseñar el código para que el motor sea intercambiable: llama-server en local, vLLM en GPU (los dos exponen una API compatible con OpenAI).
