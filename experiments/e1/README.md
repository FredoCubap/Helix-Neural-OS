# E1 — Cambio de LoRA por petición en llama-server

**Fecha:** 2026-11-06 · **Rama:** `exp/e1-lora-swap` · **Equipo:** ver `docs/hardware.md`

## Montaje

- Base: `models/qwen2.5-0.5b-instruct-q8_0.gguf` (GGUF propio de Qwen2.5-0.5B-Instruct [F-27]).
- LoRA: Python [F-29] (id 0) y SQL [F-30] (id 1), convertidos con `convert_lora_to_gguf.py --outtype f16` (17,6 MB c/u; 35 MB en fp32 original).
- Servidor (llama.cpp b12121, Vulkan, Iris Xe):
  ```
  llama-server -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
    --lora models/lora/python.gguf,models/lora/sql.gguf --lora-init-without-apply \
    -ngl 99 --host 127.0.0.1 --port 8080 -c 4096
  ```
- Cada petición elige adaptador con `"lora": [{"id": N, "scale": 1.0}]` (`[]` = base sola). Script: `compare.py`.

## Resultados — latencia (2.ª corrida, en caliente)

| Config | Prompt (~40 tokens) | Generación |
|---|---|---|
| base | 104–125 ms | 16,4–16,6 t/s |
| + LoRA Python | 297–354 ms | 16,9–17,0 t/s |
| + LoRA SQL | 330–343 ms | 17,2–17,8 t/s |

- **El cambio de adaptador entre peticiones no tiene costo de carga medible**: los dos LoRA ya están en memoria y se eligen por petición.
- **Costo real:** la lectura del prompt con LoRA aplicado es ~3× más lenta (+~200 ms en un prompt corto). La generación no cambia.
- La 1.ª petición tras arrancar tarda ~8 s (calentamiento de Vulkan); hay que descartarla al medir.
- Las diferencias de "tiempo total" en `compare.py` vienen del largo de la respuesta, no del adaptador.

## Resultados — calidad (temperatura 0)

| Pregunta | base (Q8) | + Python | + SQL |
|---|---|---|---|
| Función Python sin duplicados | Explica, **sin código** | Código correcto (`seen` + `result`) | Código correcto, con docstring |
| "How many singers?" (chat) | SQL inventado (tabla falsa) | `SELECT COUNT(*) FROM concert_singer` ✅ | SQL inventado |
| Mismo, prompt crudo (`/completion`, formato de entrenamiento del LoRA SQL) | SQL inventado | ✅ + texto extra | SQL inventado (`TESLA_FACT_SONG`) |

## Verificación de fidelidad de la conversión

Mismo prompt crudo y decodificación greedy, en Transformers + PEFT (fp32) vs. llama-server (GGUF):

- **LoRA SQL:** salida **idéntica** en PEFT y en GGUF → la conversión LoRA→GGUF es fiel; el LoRA de SQL es malo de por sí.
- **Base:** PEFT fp32 responde bien (`SELECT COUNT(*) FROM concert_singer;`); el GGUF Q8_0 inventa una consulta. → **la cuantización Q8_0 cambia la respuesta greedy en un modelo de 0,5B.**

## Conclusiones

1. El mecanismo multi-LoRA por petición funciona en local sin GPU dedicada, con costo despreciable. Lo "resuelto" de la guía se confirma también en llama.cpp.
2. **El nombre del adaptador no garantiza la especialidad**: el LoRA "Python" respondió mejor el SQL que el LoRA "SQL". Un router por descripción (como el de la guía) habría elegido el peor. → para el router hace falta evaluar los adaptadores, no confiar en su etiqueta (es lo que hace LORAUTER con conjuntos de validación [F-01]).
3. Para medir conocimiento (E2–E4), usar la base en precisión completa (Transformers) o al menos f16, no Q8_0: la cuantización mete ruido del mismo orden que el efecto que queremos medir.

## Pendiente

- Mezclar adaptadores con escala parcial (p. ej. 0,5 + 0,5) y medir.
- Medir con prompts largos (el sobrecosto de prompt escala con el largo).
- Repetir con el GGUF base en f16.
