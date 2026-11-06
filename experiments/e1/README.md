# E1 — Cambio de LoRA por petición en llama-server

**Fecha:** 2026-11-06 · **Rama:** `exp/e1-lora-swap` · **Equipo:** ver `docs/hardware.md`

## Montaje

- Base: GGUF propio de Qwen2.5-0.5B-Instruct [F-27], en dos variantes: `q8_0` (525 MB) y `f16` (994 MB).
- LoRA: Python [F-29] (id 0) y SQL [F-30] (id 1), convertidos con `convert_lora_to_gguf.py --outtype f16` (17,6 MB c/u; 35 MB en el fp32 original).
- Servidor (llama.cpp b12121, Vulkan, Iris Xe):
  ```
  llama-server -m models/qwen2.5-0.5b-instruct-<q8_0|f16>.gguf \
    --lora models/lora/python.gguf,models/lora/sql.gguf --lora-init-without-apply \
    -ngl 99 --host 127.0.0.1 --port 8080 -c 4096
  ```
- Scripts: `compare.py` (misma pregunta con base / Python / SQL) y `extra.py` (`mix`, `long`, `greedy`). Salidas en `runs/`.

## ⚠️ Trampa encontrada: `"lora": []` no apaga los adaptadores

En llama.cpp b12121, una petición con `"lora": []` **aplica todos los adaptadores con la escala global** (1.0), aunque el
servidor arranque con `--lora-init-without-apply` (`GET /lora-adapters` también muestra `scale: 1.0`). Comprobado: con `[]`
la salida es idéntica a `py 1.0 + sql 1.0` en todas las combinaciones de petición previa y caché.
**Regla:** mandar siempre la escala de *todos* los adaptadores (`scales()` en `compare.py`).

La primera versión de E1 (`runs/e1-compare-run1.txt`, `runs/e1-compare-run2.txt`) cayó en esta trampa: su "base" era en
realidad los dos LoRA juntos. Esas corridas quedan como registro, pero **sus conclusiones de calidad y de latencia de la
base no valen**. También era falsa la conclusión "Q8_0 cambia la respuesta greedy" de esa versión (ver abajo).

## Resultados (corridas válidas: `runs/q8_0-v2.txt`, `runs/f16.txt`, `runs/f16-compare.txt`)

### 1. Cambio de adaptador
No hay costo de carga: los LoRA están en memoria y cada petición fija sus escalas. Pasar de base a Python a SQL entre
peticiones consecutivas no agrega tiempo más allá del costo de cómputo de abajo.

### 2. Costo del LoRA según largo del prompt (lectura del prompt, en ms; media de 2 corridas)

| Tokens | Base Q8_0 | +1 LoRA Q8_0 | +2 LoRA Q8_0 | Base f16 | +1 LoRA f16 | +2 LoRA f16 |
|---|---|---|---|---|---|---|
| ~53 | 318 | 419 | 498 | 304 | 343 | 507 |
| 403 | 1726 | 2017 | 2416 | 1242 | 1595 | 1863 |
| 1603 | 6877 | 8353 | 9541 | 5510 | 6631 | 7732 |

- **Cada LoRA activo agrega ~20 % al tiempo de lectura del prompt**, y el costo escala linealmente con el largo
  (+~1,1–1,5 s por LoRA en 1600 tokens). Dos LoRA ≈ +40 %.
- Generación: Q8_0 ~16–17 t/s; f16 ~14 t/s. Con LoRA baja ~5 %.
- **f16 lee el prompt ~20–28 % más rápido que Q8_0** en la Iris Xe (Q8_0 gana en generación). Inesperado; se anota
  para elegir formato según el uso (prompts largos vs. respuestas largas).
- Primera petición tras arrancar: ~8 s de calentamiento de Vulkan; descartarla al medir.

### 3. Mezcla de adaptadores (SQL crudo / Python en chat; igual en Q8_0 y f16 salvo detalle)

| Config | SQL "How many singers?" | Python "sin duplicados" |
|---|---|---|
| base | `SELECT COUNT(*) FROM concert_singer;` ✅ | con código ✅ |
| py 1.0 | ✅ | con código ✅ |
| sql 1.0 | SQL inventado ❌ | con código ✅ |
| py 0.5 + sql 0.5 | `SELECT COUNT(*) AS total FROM concert_singer;` ✅ | con código ✅ |
| py 1.0 + sql 1.0 | SQL con condiciones inventadas ❌ | Q8_0: **sin código** ❌ · f16: con código |
| py 0.5 | ✅ | con código ✅ |

- Sumar adaptadores a escala completa **degrada** al modelo; a escala 0,5 + 0,5 se mantiene bien. La interferencia entre
  módulos es real y depende de la escala → hay que normalizar pesos al combinar (como la fusión ponderada de LORAUTER [F-01]).

### 4. Fidelidad: PEFT fp32 vs. GGUF Q8_0 vs. GGUF f16 (greedy, prompt crudo SQL)

| | Base | + LoRA SQL |
|---|---|---|
| Transformers + PEFT fp32 | `SELECT COUNT(*) FROM concert_singer; \n\nExplanation:...` | `...FROM TESLA_FACT_SONG f WHERE f.Song_ID` |
| GGUF Q8_0 | idéntico ✅ | idéntico ✅ |
| GGUF f16 | idéntico ✅ | difiere desde el nombre de tabla (`FROM TABLES f ...`) |

- La conversión base y LoRA a GGUF es fiel. Las diferencias son de redondeo numérico y aparecen tarde en la generación.
- En chat, la base Q8_0 y la f16 sí dieron respuestas SQL distintas (Q8_0 más larga y peor) → en un modelo de 0,5B la
  precisión puede cambiar el camino greedy. Para medir conocimiento (E2–E4) usar Transformers fp32/bf16 como referencia.

## Conclusiones

1. El multi-LoRA por petición funciona en local sin GPU dedicada. Cambiar de adaptador es gratis; el costo es de cómputo
   (~20 % más de lectura de prompt por LoRA activo, proporcional al largo).
2. **El nombre del adaptador no garantiza la especialidad.** El LoRA "SQL" es *peor* que la base en SQL; el de "Python" no
   empeora nada. Un router por descripción (como el de la guía) elegiría mal → hay que evaluar los adaptadores con datos de
   validación, no fiarse de la etiqueta.
3. **Combinar módulos tiene interferencia**: a escala completa se degradan; con pesos que suman ~1 se mantienen.
4. Las APIs pueden tener defaults engañosos (`"lora": []`): toda medición necesita un control que verifique que la
   configuración aplicada es la que creemos (aquí, comparar contra PEFT).

## Pendiente (para otros experimentos)

- Evaluación sistemática de adaptadores con un conjunto de validación (base del router, E3/Q3).
- Mezclas con más de dos adaptadores y pesos aprendidos.
