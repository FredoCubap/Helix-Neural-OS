# Hardware disponible y consecuencias

## Equipo local (medido el 2026-11-05)

| Componente | Valor |
|---|---|
| CPU | Intel Core i5-1235U (12.ª gen, 10 núcleos / 12 hilos, portátil de bajo consumo) |
| GPU | Intel Iris Xe integrada (~2 GB compartidos). **Sin GPU dedicada, sin CUDA.** |
| RAM | 39,7 GB |
| Disco libre (C:) | 343 GB |
| SO | Windows 11 Pro |
| Ollama | 0.34.1 instalado, sin modelos descargados (hay una actualización a 0.40.14 descargada, sin aplicar) |

## Qué no corre aquí

- **vLLM**: necesita GPU (CUDA/ROCm). Su backend de CPU es para Linux, no es nuestro caso. → toda la parte de servido de la guía queda fuera del equipo local.
- **QLoRA con bitsandbytes**: la cuantización 4-bit de bitsandbytes necesita CUDA. **Unsloth** también.
- **Entrenar LoRA sobre modelos de 7–8B**: imposible en tiempo razonable en CPU.

## Qué sí corre aquí

- **Inferencia de modelos cuantizados GGUF** con Ollama o llama.cpp: modelos 1–4B con fluidez, 7–8B lento pero usable (40 GB de RAM alcanzan).
- **LoRA por petición con llama-server (llama.cpp)** [F-22]: `--lora` carga varios adaptadores, `--lora-init-without-apply` los carga sin aplicar, y cada petición elige cuáles usar y con qué peso (campo `lora`, con `id` y `scale`). Esto es un **sustituto local del multi-LoRA de vLLM**, y además permite mezclar adaptadores con pesos (como en LORAUTER).
- **Ollama**: tiene que verificarse si la versión instalada sigue soportando `ADAPTER` en el Modelfile; la doc actual no lo muestra [F-23]. Aun cuando lo soporte, sería un adaptador fijo por modelo, no un cambio por petición. → se usa para pruebas rápidas del modelo base, no como motor del sistema.
- **Entrenar LoRA chicos en CPU** con PEFT + Transformers (fp32/bf16) sobre modelos de 0,5–1,5B (p. ej. Qwen2.5‑0.5B/1.5B, SmolLM2) y corpus pequeños: lento, pero alcanza para los experimentos de capacidad y de Deep Context Distillation a escala reducida (E2, E3).
- **Edición de conocimiento (EasyEdit, AlphaEdit, μKE)** sobre modelos chicos tipo GPT‑2 / 0,5B: plausible en CPU, por confirmar.

## Estrategia

1. **Local = laboratorio a escala chica.** Las preguntas de fondo (¿un LoRA guarda hechos? ¿cuánto? ¿se puede editar?) se pueden responder con modelos de 0,5–1,5B. Si no funciona en chico, no hay que gastar GPU en grande.
2. **GPU en la nube solo cuando haga falta escalar** (Colab/Kaggle gratis con T4 de 16 GB, o alquiler por hora). Antes de usarla, decidir proveedor y presupuesto con Fredo.
3. Diseñar el código para que el motor sea intercambiable: llama-server en local, vLLM en GPU (los dos exponen una API compatible con OpenAI).
