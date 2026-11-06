# E2 — ¿Puede un LoRA aprender el lore de Kenshi?

**Inicio:** 2026-11-06 · **Rama:** `exp/e2-kenshi` · **Datos:** D-01 (`data/kenshi/`, fuera de git)

Motivación de fondo: un NPC de Kenshi necesita **lore verdadero** (inteligencia cristalizada) y **voz de facción**
(perspectiva). E2 mide si un LoRA puede cargar lo primero y si lo separa de lo segundo. Destino: conectarlo a Kenshi
vía Sentient Sands [F-33], que acepta cualquier API compatible con OpenAI.

## 1. Preparación del dataset (`prepare.py`)

El JSONL viene ordenado por bloques (revisado a mano):

| Bloque | Filas | Útiles | Contenido |
|---|---|---|---|
| `lore_narrador` | 0–208 | 209 | Preguntas de lore en voz enciclopédica |
| `lore_voz` | 209–237 | 29 | Las mismas preguntas respondidas por un NPC |
| `escenas` | 238–890 | 646 (7 descartadas) | Escenas de rol; acotaciones sin marcar (p. ej. "Saca su espada lentamente ¡...!") |
| `facciones` | 891–1028 | 138 | Voz Shek / Nación Sagrada, acotaciones entre paréntesis |

- Descartadas: 7 filas donde la respuesta es copia literal de la pregunta.
- 67 preguntas aparecen más de una vez, **siempre con respuestas distintas**: son paráfrasis del mismo hecho
  (útil: sirve para entrenar con una y evaluar con otra). Algunas cruzan bloques con **perspectiva opuesta**, p. ej.
  Renacimiento = "campo de trabajo forzado" (narrador) vs. "lugar sagrado donde los pecadores purifican sus almas"
  (voz de la Nación Sagrada). Es exactamente la prueba hecho vs. perspectiva.

## 2. Línea base: la base no sabe Kenshi (`baseline.py`)

20 preguntas de `lore_narrador` (semilla 42), base f16 sin LoRA, temperatura 0. Revisión manual.

| Condición | Correctas | Comportamiento |
|---|---|---|
| Sin contexto | **0/20** | Alucina con seguridad: la Cruzada Sagrada "fundada por Pedro II de Castilla en 1497", Cat-Lon "ciudad de Guizhou, China", la Rebelión Roja "de Franco", el hachís "un arma de fuego". O rechaza ("temas sensibles", "contenido sexual"). |
| Con system prompt "Responde sobre el videojuego Kenshi" | **0/20** | Reconoce que es un juego de Lo-Fi Games, pero inventa todo ("las Tierras Cenicientas son una serie de juegos de rol"; "Garru, una especie de ser humano") o rechaza. |

→ Todo acierto posterior se puede atribuir al LoRA. Además, la base **alucina en vez de decir "no sé"**: en E2 hay que
medir también si el LoRA reduce o aumenta la alucinación fuera de lo entrenado (Gekhman et al. [F-12]).

## 3. Banco de evaluación (`eval_bank.jsonl`, `eval.py`)

72 preguntas propias + 47 "literales" (la pregunta exacta del dataset): paráfrasis de hechos entrenados (47), holdout
(9 hechos quitados de todo el entrenamiento con regex `anchor`), inexistentes (6), conocimiento general (6) y
perspectiva (3: pregunta neutral, no debe salir la voz de facción). Puntuación por claves sin tildes, con:
- **autoprueba** (`--selftest`): cada clave debe estar en las filas fuente del dataset;
- **control de eco**: una clave que ya está en la pregunta no cuenta (la base "acertaba" 10/47 literales repitiendo
  "Tengu" antes de inventar que era "de One Piece, de J.K. Rowling").

## 4. Entrenamiento (Colab T4, `colab_train.ipynb`)

Mismo `train_lora.py` que en local (CPU: ~46 tok/s; T4: ~2.500–3.000 tok/s, ~55× más rápido). r=16, α=32, 7 módulos,
lr 2e-4 coseno, bs 8, 8 épocas, pérdida solo en respuestas. torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2.

| Variante | Ejemplos | Tokens | Loss época 1 → 8 | Tiempo T4 |
|---|---|---|---|---|
| lore | 197 | 28,7 k | 2,35 → 0,10 | 1,2 min |
| voces | 808 | 85,7 k | 2,67 → 0,16 | 4,7 min |
| todo | 1005 | 114,4 k | 2,55 → 0,14 | 6,1 min |

Conversión a GGUF verificada: PEFT y llama-server coinciden token a token hasta que el redondeo los separa.

## 5. Resultados (llama-server, base f16, escalas explícitas; control base0 = idéntico a la línea base)

| | base | lore | voces | todo |
|---|---|---|---|---|
| literal (37 puntuables) | 0 % | 24 % | 3 % | 30 % |
| **paráfrasis (47)** | 0 % | **26 %** | 4 % | **26 %** |
| holdout (9) | 0 % | 0 % | 0 % | 0 % |
| **perspectiva neutral (3)** | 0/3 | **3/3** | 0/3 | **0/3** |
| no inventa (6) | 3/6 | 1/6 | 0/6 | 0/6 |
| conocimiento general (6) | 4/6 | 3/6 | 1/6 | 4/6 |

Revisión manual (≈50 respuestas):
1. **Aprende la forma, no los hechos.** Con loss 0,10 reproduce el estilo de las respuestas, pero **recombina
   entidades**: "la líder de los Ninjas Flotsam es Saitou", "Katanas, con una esposa llamada Moll", "la capital del Reino
   Shek es Sihangi / Kral", "el líder de los Antiesclavistas es Venganza". Incluso preguntas *literales* fallan ~75 %.
   El 26 % es una **cota superior**: la puntuación por claves cuenta aciertos aunque vengan con invenciones al lado
   ("liderada por Tinfist… su líder actual es Stilt").
2. **Aumenta la alucinación** (Gekhman et al. [F-12] confirmado): ningún LoRA dice "no sé"; inventan con seguridad sobre
   el "Rey Murciélago" o sitúan El Cubo (holdout) "en el Pantano".
3. **Daño fuera de dominio**: todo se vuelve Kenshi ("12 por 12 es igual a 12.000 cats", "el planeta rojo es la Nación
   Sagrada") y aparece **texto en chino** a mitad de respuesta (lore, voces y todo) → lr 2e-4 × 8 épocas es demasiado.
   El LoRA de voces es el peor (general 1/6).
4. **Separar módulos importa (resultado a favor de la tesis de Helix):** el LoRA de *lore* responde neutral (3/3);
   el de *todo* mete la voz de facción en preguntas neutrales (0/3: "(Con desprecio) Matar a un personaje es...").
   Mezclar hechos y voces en un solo módulo contamina; mantenerlos separados y combinarlos con escalas (E1) es el camino.

## 6. Próximos pasos

1. **Barrido de épocas/lr** (2, 4, 8 épocas; lr 1e-4) para medir el compromiso aprendizaje vs. daño. Minutos en T4.
2. **Aumento de conocimiento**: varias paráfrasis por hecho en el entrenamiento (hipótesis de *Physics of LMs* sobre
   almacenamiento vs. extracción, por verificar en la fuente) — es la palanca más prometedora contra la recombinación.
3. **E3: RAG vs. LoRA vs. LoRA+RAG** con el mismo banco — el estudio de PRAG [F-10] sugiere que el texto en contexto
   sigue haciendo falta.
4. **Base más grande** (Qwen2.5-1.5B; luego Qwen3-4B, el candidato de Fredo para el juego) para ver el efecto de capacidad.
5. **Lore + voz por separado combinados por escala** (p. ej. lore 1.0 + voces 0.3) como prototipo de NPC.

Salidas crudas en `data/kenshi/runs/eval_e2_*.jsonl` (fuera de git: contienen texto del dataset).
