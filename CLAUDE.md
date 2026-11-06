# Helix Neural OS — instrucciones del proyecto

Proyecto de investigación: sistema operativo neural modular (núcleo de razonamiento + módulos de conocimiento). Ver `docs/`.

- **Toda fuente consultada se registra en `docs/fuentes.md`** (también las descartadas y las búsquedas sin resultado), con tipo (primaria/secundaria), nivel de lectura y fecha. Las afirmaciones en los documentos se citan con su ID `[F-NN]`.
- **Hardware local sin GPU dedicada** (ver `docs/hardware.md`): no proponer nada que necesite CUDA para correr en local. El uso de GPU en la nube se decide con Fredo.

## Entorno

- **Dependencias siempre en el entorno virtual del proyecto** (`.venv/`, creado con `uv`), nunca en el Python global. Instalar con `uv pip install -p .venv ...`.
- Dependencias fijadas por perfil en `requirements/` (`cpu.txt`; `cuda.txt` cuando toque).
- Herramientas externas en `.tools/` (p. ej. `llama.cpp` en la misma build que el binario instalado, para `convert_hf_to_gguf.py` y `convert_lora_to_gguf.py`). Modelos en `models/`. Ambas carpetas fuera de git.

## Ramas y hardware

- **`main` es la rama de investigación** y la única permanente: documentos, código y resultados.
- **El hardware es configuración, no rama**: perfiles (`configs/cpu.yaml`, `configs/cuda.yaml`) y dependencias opcionales por perfil. Motor local: `llama-server` (llama.cpp); motor GPU: vLLM. Ambos exponen API compatible con OpenAI.
- **Ramas cortas** por experimento o cambio (`exp/e1-lora-swap`, `feat/router`), fusionadas a `main` al cerrar. Versiones estables con tags (`v0.1`).

## Modelo de laboratorio

- Base: **Qwen2.5-0.5B-Instruct**; validación: **Qwen2.5-1.5B-Instruct**. Transformer denso estándar (compatible con herramientas de edición e interpretabilidad, LoRA convertible a GGUF). Alternativa densa: Qwen3-0.6B/1.7B. Qwen3.5 (híbrido Gated DeltaNet) solo como comparación. Ver `docs/hardware.md`.
