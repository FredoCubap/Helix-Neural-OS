# Registro de fuentes

Toda fuente consultada en el proyecto se anota aquí, aunque se haya descartado.

- **Tipo:** `P` = primaria (paper, documentación oficial, repositorio); `S` = secundaria (resumen, blog, agregador, prensa).
- **Nivel de lectura:** `abstract` / `resumen` (solo snippet o resumen) / `parcial` / `completo`.
- **Fecha** = día de la consulta. Usar el formato `[F-NN]` para citar desde otros documentos.

## Papers y documentación técnica

| ID | Fuente | Autores / año | URL | Tipo | Lectura | Fecha | Usado en |
|---|---|---|---|---|---|---|---|
| F-01 | Effective LoRA Adapter Routing using Task Representations (LORAUTER) | Dhasade et al., 2026 | https://arxiv.org/abs/2601.21795 | P | abstract | 2026-11-05 | mapa §2, §3.2 |
| F-02 | μKE: Matryoshka Unstructured Knowledge Editing of LLMs | Su, Huang, Zhang, Zhang — COLM 2025 | https://arxiv.org/abs/2504.01196 · código: https://github.com/PurCL/muke | P | abstract | 2026-11-05 | mapa §2, §3.5 |
| F-03 | MoNe: Modular Neural Memory for Efficient Long Context Inference | (autores sin verificar) | https://arxiv.org/pdf/2608.17616 | P | resumen | 2026-11-05 | mapa §2, §3.4 |
| F-04 | Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization | Guo, Syed, Sheshadri, Ewart, Dziugaite — ICML 2025 | https://arxiv.org/abs/2410.12949 · https://icml.cc/virtual/2025/poster/46254 | P | abstract | 2026-11-05 | mapa §2, §3.5 |
| F-05 | Does Localization Inform Unlearning? | Lee, Hwang, Lim, Kim, 2025 | https://arxiv.org/abs/2505.16252 | P | resumen | 2026-11-05 | mapa §2, §3.5 |
| F-06 | Training Plug-n-Play Knowledge Modules with Deep Context Distillation | Caccia, Ansell, Ponti, Vulić, Sordoni — COLM 2025 | https://arxiv.org/abs/2503.08727 | P | abstract | 2026-11-05 | mapa §3.3 |
| F-07 | Dynamic Parametric RAG for Test-time Knowledge Enhancement (DyPRAG) | 2025 | https://arxiv.org/abs/2503.23895 | P | resumen | 2026-11-05 | mapa §3.3 |
| F-08 | Poly-PRAG | 2025 | https://arxiv.org/abs/2511.17044 | P | resumen | 2026-11-05 | mapa §3.3 |
| F-09 | DistilledPRAG: Privacy-Preserving Reasoning with Knowledge-Distilled Parametric RAG | 2025 | https://arxiv.org/abs/2509.01088 | P | resumen | 2026-11-05 | mapa §3.3 |
| F-10 | The Role of Parametric Injection — A Systematic Study of Parametric RAG | 2025 | https://arxiv.org/abs/2510.12668 | P | resumen | 2026-11-05 | mapa §3.3 |
| F-11 | Memory Layers at Scale | Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh — Meta FAIR, 2024 | https://arxiv.org/abs/2412.09764 · código: https://github.com/facebookresearch/memory | P | abstract | 2026-11-05 | mapa §3.4 |
| F-12 | Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? | Gekhman et al. — EMNLP 2024 | https://arxiv.org/abs/2405.05904 | P | abstract | 2026-11-05 | mapa §3.3 |
| F-13 | AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models | ICLR 2025 | https://arxiv.org/abs/2410.02355 | P | abstract | 2026-11-05 | mapa §3.5 |
| F-14 | WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of LLMs | NeurIPS 2024 | https://arxiv.org/abs/2405.14768 | P | abstract | 2026-11-05 | mapa §3.4 |
| F-15 | EasyEdit (framework de edición de conocimiento) | zjunlp | https://github.com/zjunlp/EasyEdit | P | resumen | 2026-11-05 | mapa §3.5 |
| F-16 | S-LoRA: Serving Thousands of Concurrent LoRA Adapters | Sheng et al., 2023 | https://arxiv.org/abs/2311.03285 | P | abstract | 2026-11-05 | mapa §2, §3.1 |
| F-17 | Punica: Multi-Tenant LoRA Serving | Chen et al., 2023 | https://arxiv.org/abs/2310.18547 | P | abstract | 2026-11-05 | mapa §2, §3.1 |
| F-18 | Towards Modular LLMs by Building and Reusing a Library of LoRAs (Arrow, MBC) | Ostapenko et al. — ICML 2024 | https://arxiv.org/abs/2405.11157 · https://proceedings.mlr.press/v235/ostapenko24a.html | P | abstract | 2026-11-05 | mapa §3.2 |
| F-19 | Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws | Allen-Zhu, Li — ICLR 2025 | https://arxiv.org/abs/2404.05405 | P | abstract | 2026-11-05 | mapa §3.3 |
| F-20 | vLLM — LoRA Adapters (documentación) | vLLM | https://docs.vllm.ai/en/latest/features/lora.html | P | parcial | 2026-11-05 | mapa §2, §3.1 |
| F-21 | PEFT — Hotswapping adapters (documentación) | Hugging Face | https://huggingface.co/docs/peft/package_reference/hotswap | P | resumen | 2026-11-05 | mapa §3.1 |
| F-22 | llama.cpp server README (flags `--lora`, `--lora-scaled`, `--lora-init-without-apply`, campo `lora` por petición) | ggml-org | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | P | parcial | 2026-11-05 | hardware.md |
| F-23 | Ollama — Modelfile (documentación); la versión actual no muestra la instrucción `ADAPTER` | Ollama | https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx | P | parcial | 2026-11-05 | hardware.md |
| F-24 | Qwen3.5-0.8B — ficha oficial (arquitectura 6 × (3 × Gated DeltaNet + 1 × Gated Attention), 24 capas) | Qwen | https://huggingface.co/Qwen/Qwen3.5-0.8B | P | resumen | 2026-11-05 | CLAUDE.md, hardware.md |
| F-25 | easyeditor en PyPI (soporte de Qwen desde oct-2023; versiones Qwen2.5/Qwen3 sin confirmar) | zjunlp | https://pypi.org/project/easyeditor/ | P | resumen | 2026-11-05 | hardware.md |
| F-26 | llama.cpp en winget (`ggml.llamacpp`, versión b12121) | ggml-org | https://github.com/ggml-org/llama.cpp | P | parcial | 2026-11-05 | hardware.md |

## Fuentes secundarias (contexto; no citar como evidencia)

| ID | Fuente | URL | Lectura | Fecha | Nota |
|---|---|---|---|---|---|
| S-01 | Resumen de LORAUTER (catalyzex) | https://www.catalyzex.com/paper/effective-lora-adapter-routing-using-task | resumen | 2026-11-05 | Cifras "101,2 % del Oracle" y "+5,2 pts" vienen de aquí, sin verificar en el PDF. |
| S-02 | Modelwire: memoria modular reduce 80 % el costo del contexto largo (MoNe) | https://themodelwire.com/article/modular-memory-layer-cuts-long-context-inference-costs-by-80-percent-01M0BVAZXF8S35J97VEMY38WJQ | resumen | 2026-11-05 | Cifra del 80 % sin verificar. |
| S-03 | Noticias de la compra de Modular por Qualcomm (nand-research, noqta) | https://www.nand-research.com/qualcomm-acquires-modular-for-its-hardware-agnostic-ai-software-layer/ | resumen | 2026-11-05 | Probable origen de la confusión "MoNe = Qualcomm" de la guía. |
| S-04 | Reseña de Mechanistic Unlearning (The Moonlight) | https://www.themoonlight.io/review/mechanistic-unlearning-robust-knowledge-unlearning-and-editing-via-mechanistic-localization | resumen | 2026-11-05 | |
| S-05 | VentureBeat sobre Memory Layers | https://venturebeat.com/ai/meta-proposes-new-scalable-memory-layers-that-improve-knowledge-reduce-hallucinations/ | resumen | 2026-11-05 | |
| S-06 | Solenya: "Cognitive core" (opinión sobre la idea de Karpathy) | https://solenya.ai/blog/22-cognitive-core | resumen | 2026-11-05 | No es fuente primaria de Karpathy; falta encontrar la original. |
| S-07 | killerstorm: Cognitive core | https://killerstorm.github.io/ai/2025/11/04/cogcore.html | resumen | 2026-11-05 | |
| S-08 | Preprint sobre benchmarks de edición (INLAY) | https://arxiv.org/abs/2608.26292 | resumen | 2026-11-05 | Reporta WISE débil y AlphaEdit fuerte en EasyEdit; sin revisión por pares. |
| S-09 | Ertas AI: mejores LLM chicos para local 2026 | https://www.ertas.ai/best/best-small-llm-for-local-deployment | resumen | 2026-11-05 | Ranking tipo SEO. |
| S-10 | PromptQuorum: modelos locales chicos 2026 | https://www.promptquorum.com/local-llms/small-local-llm-models | resumen | 2026-11-05 | Ranking tipo SEO. |
| S-11 | localaimaster: guía de modelos chicos 2026 | https://localaimaster.com/blog/small-language-models-guide-2026 | resumen | 2026-11-05 | Ranking tipo SEO. |
| S-12 | trilogyai: Deep dive de Qwen 3.5 | https://trilogyai.substack.com/p/deep-dive-qwen-35-brings-native-multimodality | resumen | 2026-11-05 | Confirma patrón 3:1 DeltaNet/atención. |

## Buscado y no encontrado

| Qué | Fecha | Resultado |
|---|---|---|
| Formato ".SBF" (semantic block format) para LoRA | 2026-11-05 | Sin resultados; se considera sin respaldo. |
| Vínculo entre MoNe y Qualcomm | 2026-11-05 | No encontrado. |
| Afirmación "LoRA learns less, forgets less" | 2026-11-05 | Pendiente; no apareció en la búsqueda. |
| Soporte de Qwen2.5/Qwen3 en EasyEdit y en TransformerLens | 2026-11-05 | No confirmado en la documentación; probar en E1/E4. |
