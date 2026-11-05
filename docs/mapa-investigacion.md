# Mapa de investigación — Helix Neural OS

> Estado: v0, 2026-11-05. Punto de partida: `docs/guia-arquitectura-neural-os.md`.
> Objetivo de este documento: separar lo que **ya existe y funciona**, lo que **es investigación abierta**
> y lo que **la guía afirma sin respaldo**, para decidir dónde experimentar primero.
> Todas las fuentes están registradas en `docs/fuentes.md`; el hardware y sus límites, en `docs/hardware.md`.

## 1. La hipótesis central (y por qué es difícil)

La guía propone separar **inteligencia fluida** (núcleo pequeño que razona) de **inteligencia cristalizada**
(conocimiento empaquetado en adaptadores LoRA que se inyectan en caliente). Es la misma idea que Karpathy llama
*cognitive core*: un modelo de ~1B que conserva los algoritmos de razonamiento y busca los hechos afuera.

El problema duro **no es servir LoRAs** (eso está resuelto). El problema duro es:

1. **¿Un LoRA guarda conocimiento factual de verdad, o solo estilo/formato?** La evidencia es mixta (ver §3.3).
2. **¿Se puede quitar conocimiento a un modelo sin quitarle razonamiento?** Nadie ha mostrado un núcleo así a escala.
3. **¿Cómo componer varios módulos a la vez sin que se interfieran?**
4. **¿Cómo editar/borrar un hecho dentro de un módulo sin romper el resto?**

## 2. Verificación de las afirmaciones de la guía

| Afirmación de la guía | Estado | Nota |
|---|---|---|
| vLLM sirve varios LoRA sobre un modelo base compartido | ✅ Existe | `--enable-lora`, `--lora-modules nombre=ruta`, `--max-loras`, `--max-cpu-loras`, carga en caliente con `/v1/load_lora_adapter` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (la doc advierte riesgo de seguridad). |
| Caché LRU y batching heterogéneo de LoRAs | ✅ Existe | Viene de S-LoRA (Unified Paging) y Punica (kernels SGMV). |
| LORAUTER (enrutado por representación de tareas) | ✅ Existe | arXiv 2601.21795 (ene-2026). Sin entrenar el router; probado con >1.500 adaptadores. **Ojo:** el original fusiona top-K adaptadores; la guía solo elige 1 por similitud coseno, que es una versión mucho más simple. |
| μKE (Matryoshka Unstructured Knowledge Editing) | ✅ Existe | arXiv 2504.01196, COLM 2025, código en `PurCL/muke`. Edita el **modelo**, no un adaptador LoRA — la guía no explica cómo se combinaría con módulos. |
| Mechanistic Localization para desaprender | ✅ Existe | Guo et al., ICML 2025 (arXiv 2410.12949). Contrapunto: "Does Localization Inform Unlearning?" (arXiv 2505.16252) dice que no es tan nítido. |
| MoNe "desarrollado por Qualcomm" | ⚠️ Atribución dudosa | El paper MoNe (arXiv 2608.17616) existe — memoria neural *fast-weight* por capa para contexto largo — pero no encontramos vínculo con Qualcomm. Además MoNe es memoria de **contexto**, no de conocimiento de dominio. |
| Formato de bloques semánticos ".SBF" | ❌ Sin fuente | No aparece en ninguna búsqueda. Tratar como inventado hasta encontrar origen. |
| Adaptador de ~25 MB con r=16 en los 7 módulos | ❌ Incorrecto | Cálculo para Llama‑3.1‑8B (32 capas, hidden 4096, KV 1024, MLP 14336), r=16 en q,k,v,o,gate,up,down: ≈ 1,31 M params/capa × 32 ≈ **42 M params ≈ 84 MB en fp16**. ~25 MB solo cuadra con r=16 en q,k,v,o. |
| Cambio de adaptador en 10–50 ms / <0,5 s | 🔍 Medir | Plausible si ya está en caché CPU; hay que medirlo nosotros. |
| Reducción de costo 60–80 % | 🔍 Sin fuente | Número de marketing; no citar. |
| Edición de "lookup tables" sin tocar razonamiento | 🔍 Abierto | Es investigación activa, no una capa de software lista. |

## 3. Mapa de literatura por área

### 3.1 Servido multi-LoRA (resuelto — usar, no investigar)
- **S-LoRA** — arXiv 2311.03285. Miles de adaptadores, paginación unificada adaptadores+KV cache.
- **Punica** — arXiv 2310.18547. Kernels para batch de LoRAs distintos (~2 ms/token extra).
- **vLLM LoRA** — https://docs.vllm.ai/en/latest/features/lora.html
- **PEFT hotswap** — https://huggingface.co/docs/peft/package_reference/hotswap (mismos módulos objetivo y alpha).

### 3.2 Enrutado y composición de módulos
- **LORAUTER** — arXiv 2601.21795. Embeddings de tarea + fusión ponderada top-K en el espacio de salida.
- **Arrow + MBC** (Ostapenko et al., ICML 2024) — arXiv 2405.11157. Librería de LoRAs, enrutado zero-shot por adaptador.
- Pendiente de revisar: LoraHub, X-LoRA, MoLE, LoRA Soups, TIES/DARE merging.

### 3.3 ¿Puede un LoRA almacenar conocimiento? (el corazón del proyecto)
- **Knowledge Modules + Deep Context Distillation** (Caccia, Sordoni et al., COLM 2025) — arXiv 2503.08727.
  Documento → LoRA. Clave: *next-token prediction es mal objetivo*; funciona mejor destilar hidden states y logits
  de un maestro que ve el documento en contexto. Funciona **junto** a RAG, no en vez de.
- **Parametric RAG** y derivados: DyPRAG (2503.23895, hipernetwork documento→parámetros),
  Poly-PRAG (2511.17044, pocos LoRA latentes compartidos), DistilledPRAG (2509.01088).
- **Estudio sistemático de PRAG** — arXiv 2510.12668: los documentos parametrizados capturan **solo parte** de la semántica;
  recomiendan combinarlos con el texto. ⚠️ Esto contradice la idea de "liberar el contexto" de la guía.
- **Gekhman et al., EMNLP 2024** — arXiv 2405.05904: afinar con conocimiento nuevo se aprende lento y **aumenta la alucinación**.
- **Capacidad**: Allen-Zhu & Li, *Physics of LMs 3.3* (ICLR 2025, arXiv 2404.05405): ~**2 bits/parámetro**, con ~1000 exposiciones por hecho.
  Cota teórica para nuestro LoRA de 42 M params: ≈ 84 Mbit ≈ **10 MB de hechos puros** en condiciones ideales.

### 3.4 Alternativas arquitectónicas al LoRA como memoria
- **Memory Layers at Scale** (Meta FAIR) — arXiv 2412.09764, código `facebookresearch/memory`. Key-value entrenable que reemplaza
  FFN; gana fuerte en tareas factuales. Candidato serio a "inteligencia cristalizada" nativa.
- **MoNe** — arXiv 2608.17616. Memoria fast-weight por capa (para contexto largo).
- **WISE** (NeurIPS 2024) — arXiv 2405.14768: memoria principal + memoria lateral con router. Es literalmente "núcleo + módulo de conocimiento".

### 3.5 Edición y desaprendizaje
- **μKE** — arXiv 2504.01196, `github.com/PurCL/muke`.
- **AlphaEdit** (ICLR 2025) — arXiv 2410.02355: proyección al espacio nulo del conocimiento preservado.
- **Mechanistic Unlearning** — arXiv 2410.12949 (y la crítica 2505.16252).
- **EasyEdit** — `github.com/zjunlp/EasyEdit`: framework con ROME, MEMIT, WISE, AlphaEdit… punto de entrada práctico.

### 3.6 El núcleo de razonamiento ("cognitive core")
- Idea de Karpathy (sin paper formal). Pendiente: buscar trabajos sobre destilar razonamiento a modelos chicos
  separando conocimiento (p. ej. modelos 1–4B de razonamiento, *knowledge-free reasoning*).

## 4. Preguntas de investigación propias

- **Q1.** Con un modelo base chico (0,5–1,5B en CPU local), ¿un LoRA entrenado con Deep Context Distillation sobre un corpus
  de dominio responde preguntas *closed-book* mejor que el base? ¿Cuánto peor que RAG?
- **Q2.** ¿Cuánto conocimiento cabe en un LoRA según rango (r = 8, 16, 64) antes de saturar? (comparar con la cota de 2 bits/param)
- **Q3.** ¿Enrutado de 1 adaptador (coseno) vs. top-K fusionado (LORAUTER) vs. Arrow: qué gana con 5–20 módulos propios?
- **Q4.** ¿Se puede editar un hecho dentro de un LoRA (no del base) con AlphaEdit/μKE, o hay que reentrenar el módulo?
- **Q5.** ¿Memory layers / WISE son mejor sustrato que LoRA para la memoria cristalizada?

## 5. Experimentos propuestos (orden sugerido)

| # | Experimento | Responde | Costo |
|---|---|---|---|
| E0 | ✅ Inventario de hardware: sin GPU dedicada, 40 GB de RAM (ver `hardware.md`). Modelo base de laboratorio: 0,5–1,5B | Restricciones | Hecho |
| E1 | llama-server (llama.cpp) + 2 LoRAs GGUF + router coseno; medir latencia del cambio de adaptador por petición (vLLM queda para GPU en la nube) | Valida §2 (lo "resuelto") | Bajo |
| E2 | Corpus de hechos sintéticos (estilo Physics of LMs) → LoRA con next-token vs. con Deep Context Distillation | Q1, Q2 | Medio |
| E3 | Mismo corpus: LoRA vs. RAG vs. LoRA+RAG | Q1, lado "liberar contexto" | Medio |
| E4 | Editar 10 hechos con EasyEdit (AlphaEdit) en el base y en base+LoRA fusionado | Q4 | Medio |
| E5 | Prototipo de memory layer / WISE como alternativa | Q5 | Alto |

## 6. Pendientes de esta revisión

- Leer los PDFs completos (hasta ahora: abstracts y resúmenes).
- Encontrar el origen de ".SBF" o descartarlo.
- Revisar la compatibilidad vLLM + LoRA + base cuantizado (bitsandbytes / AWQ) en la versión que usemos.
