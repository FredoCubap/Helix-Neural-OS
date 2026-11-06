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

## 3. Plan

1. **Banco de evaluación:** hechos de `lore_narrador`; para los que tienen paráfrasis, entrenar con unas y evaluar con
   otras; para el resto, escribir paráfrasis nuevas. Más preguntas *fuera* del dataset (para medir alucinación).
2. **LoRA a comparar:** (a) solo lore, (b) solo escenas + facciones, (c) todo.
3. **Métricas:** acierto factual en paráfrasis, alucinación fuera de dominio, confusión hecho/perspectiva.
4. Control sintético (E2b) si E2a da resultados ambiguos.

Salidas crudas en `data/kenshi/runs/` (fuera de git: contienen texto del dataset).
