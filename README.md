# Re-ranking de noticias mediante un sistema de clasificación ideológica para mitigar las burbujas informativas
**Trabajo de Fin de Máster** : Detección automática del sesgo ideológico en artículos de seis medios de comunicación españoles mediante embeddings, clustering temático, etiquetado LLM y fine-tuning con LoRA.

---

## Descripción

Este TFM construye un sistema completo de clasificación ideológica (IZQUIERDA · CENTRO · DERECHA) sobre un corpus de **241 240 artículos** extraídos de seis medios de prensa española: El País, El Mundo, El Confidencial, elDiario.es, Público y OKDiario, cubriendo los temas de política, economía y opinión a lo largo de los últimos tres años.

El pipeline va desde el scraping automatizado hasta la inferencia final sobre el corpus completo, pasando por la construcción de un ground truth con LLMs, modelos supervisados y semi-supervisados, y ocho variantes de fine-tuning LoRA sobre Qwen3-4B.

**Modelo ganador:** `LoRA QKV r8` (Qwen3-4B + LoRA sobre módulos QKV, r=8) — F1-macro en test = **0.8303**, el único modelo con Δ val→test positivo (+0.009).

---

## Pipeline completo

```
scrapers/          → 0. Scraping de los 6 medios
notebooks/
  1_limpieza       → 1. Limpieza y normalización del corpus
  2_eda            → 2. Análisis exploratorio (EDA)
  3_embedings      → 3. Embeddings con Qwen3-0.6B (1 024 dims)
  4_clustering     → 4. Clustering temático con BERTopic + DBSCAN
  5.0_ground_truth → 5. Creación del ground truth (etiquetado LLM)
  5.1_supervisado  → 6. Modelos supervisados y semi-supervisados
  5.2_finetuning/  → 7. Fine-tuning LoRA (8 variantes, en Colab)
  5.2_zeroshot     → 8. Experimentos zero-shot y few-shot
  5.3_comparativa  → 9. Comparativa final sobre el test set
  6_inferencia     → 10. Inferencia sobre los 241 240 artículos
  7.0_dataframe    → 11. Construcción del DataFrame final
  7.1_reranking    → 12. Reranking de artículos por relevancia
```

---

## Fases del proyecto

### 0 · Scraping

El módulo `scrapers/` contiene un scraper independiente por cada medio (RSS + parsing HTML). El punto de entrada es `scrapers/run.py`, que orquesta todos los medios con soporte de checkpoints para reanudar descargas interrumpidas.

```bash
# Todos los medios (por defecto: últimos 3 años)
python scrapers/run.py

# Solo algunos medios
python scrapers/run.py --medios eldiario elpais

# Reanudar desde checkpoint
python scrapers/run.py --reanudar

# Solo mergear JSONs ya descargados
python scrapers/run.py --merge
```

Temas recopilados: `politica`, `economia`, `opinion`. Los artículos se guardan por medio en `output/<medio>.json` y se combinan en `output/TODOS.json`.

### 1 · Limpieza del corpus

`notebooks/1_limpieza.ipynb` — normalización de texto, deduplicación, filtrado de artículos demasiado cortos o en otro idioma, y guardado en `data/processed/corpus_limpio_filtrado.parquet` (241 240 artículos).

### 2 · Análisis exploratorio (EDA)

`notebooks/2_eda.ipynb` — distribución por medio, tema y fecha; análisis de longitud; detección de spikes de actividad; coautoría; drift temporal; análisis de idiomas y missingness. Figuras exportadas en `output/eda_meta_*.png`.

### 3 · Embeddings

`notebooks/3_embedings.ipynb` — generación de embeddings de 1 024 dimensiones con `Qwen/Qwen3-0.6B` sobre la concatenación `titular + abstract`. Los vectores se guardan por medio en `embeddings/*.npy` y se reducen a 2D/10D con UMAP para visualización y clustering.

### 4 · Clustering temático

`notebooks/4_clustering.ipynb` — clustering con **BERTopic** (c-TF-IDF + representación LLM de tópicos) y refinamiento de outliers con **DBSCAN** (hiperparámetros optimizados con Optuna). Se obtienen ~29 macrotemas que se visualizan en `output/clustering_6medios/`. Los tópicos permiten analizar la agenda temática diferencial de cada medio.

### 5 · Creación del ground truth

`notebooks/5.0_ground_truth.ipynb` y el módulo `ground_truth/` implementan el pipeline de etiquetado automático mediante un **panel de tres LLMs árbitros**:

| Modelo | F1 (Kennedy et al. 2025) | Sesgo conocido |
|---|---|---|
| Claude Sonnet 4.6 | 0.462 | center-collapse |
| GPT-5.2 | 0.448 | center-collapse |
| Llama 3.1 8B (Groq) | 0.397 | sesgo LEFT |

El módulo `ground_truth/voting.py` implementa un **sistema de votación ponderada** que tiene en cuenta los sesgos sistemáticos de cada modelo para resolver empates y asignar niveles de confianza (ALTA / MEDIA / BAJA). Los artículos con desacuerdo total se descartan (`DESCARTAR`) y los casos ambiguos se marcan para revisión (`REVISAR`).

El resultado es `data/ground_truth/ground_truth_clean.parquet` con ~15 000 artículos etiquetados, de los cuales los marcados como `INCLUIR` forman el ground truth de entrenamiento.

```bash
# Paso 1: muestreo estratificado (genera articulos_15000.csv)
python -m ground_truth.pipeline sample

# Paso 2: clasificar con los 3 LLMs
python -m ground_truth.pipeline
```

### 6 · Sistema de clasificación

#### 6a · Modelos supervisados y semi-supervisados (`notebooks/5.1_supervisado_semisupervisado.ipynb`)

Sobre los embeddings de 1 024 dims y splits train/val/test guardados en `data/splits/`:

| Modelo | Descripción |
|---|---|
| TF-IDF + LR | Regresión logística sobre bag-of-words |
| TF-IDF + SVM | SVM lineal sobre bag-of-words |
| MLP supervisado | `Linear → BatchNorm1d → ReLU → Dropout` × N capas |
| MLP self-training | MLP con etiquetado iterativo de no etiquetados |
| Label Propagation | Propagación de etiquetas sobre grafo de similitud |
| Label Spreading | Variante regularizada de Label Propagation |

Modelos guardados en `output/clasificacion_supervisada/`. Hiperparámetros optimizados con Optuna.

#### 6b · Experimentos zero-shot y few-shot (`notebooks/5.2_zeroshot_fewshot.ipynb`)

Evaluación de Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B y XLM-RoBERTa-XNLI sin entrenamiento. Resultados en `output/zeroshot_results.csv` y `output/fewshot_results.csv`.

#### 6c · Fine-tuning LoRA en Colab (`notebooks/5.2_finetuning_colab/`)

Ocho variantes de fine-tuning sobre **Qwen3-4B** con `AutoModelForSequenceClassification` + PEFT/LoRA, ejecutadas en Google Colab con GPU A100:

| Variante | Módulos LoRA | Rango |
|---|---|---|
| **LoRA QKV r8** (ganador) | q_proj, k_proj, v_proj | r=8 |
| LoRA QV r8 | q_proj, v_proj | r=8 |
| LoRA Regularized r8 | q_proj, v_proj | r=8 + WD |
| LoRA MLP-head r32 | q_proj, v_proj + head MLP | r=32 |
| LoRA All-linear r8 | todos los módulos lineales | r=8 |
| LoRA MLP-head r8 | q_proj, v_proj + head MLP | r=8 |
| LoRA Full-7layers r8 | 7 últimas capas | r=8 |
| QLoRA QKV r8 | q_proj, k_proj, v_proj (4-bit) | r=8 |

### 7 · Comparativa de modelos

`notebooks/5.3_comparativa_test_modelos.ipynb` — evaluación de todos los modelos sobre el mismo test set (238 artículos). Métricas: F1-macro, F1 por clase, matrices de confusión, curvas de calibración, análisis por medio y longitud. Resultados en `output/finetuning_test_metrics.csv`.

### 8 · Inferencia

`notebooks/6_inferencia_colab.ipynb` — aplicación del modelo ganador (`qwen3_4b_lora_r8_qkv`) sobre los ~238 862 artículos no etiquetados del corpus completo. Genera predicciones con probabilidades por clase (`p_izquierda`, `p_centro`, `p_derecha`) y nivel de confianza. Resultado: `output/inferencia_ideologica.csv`.

### 9 · DataFrame final y reranking

`notebooks/7.0_dataframe_final.ipynb` — construye `data/final/corpus_maestro.parquet` uniendo ground truth (etiquetas gold) con inferencias (etiquetas del modelo). La columna `label_source` indica la procedencia de cada etiqueta.

`notebooks/7.1_reranking.ipynb` — sistema de reranking de artículos por relevancia ideológica dentro de cada macrotema (ej. Economía & Social), con análisis de convergencia y comparativa baseline. Visualizaciones en `output/reranking/`.

---

## Estructura del proyecto

```
TFM/
├── config.py                      # Configuración global (rutas, delays, API keys via .env)
├── requirements.txt               # Dependencias del scraper
├── .env.example                   # Plantilla de variables de entorno (copiar a .env y rellenar)
│
├── scrapers/                      # Módulo de scraping
│   ├── run.py                     # Punto de entrada
│   ├── utils.py                   # Checkpoints, guardado, utilidades
│   └── medios/                    # Un scraper por medio
│       ├── elpais.py
│       ├── elmundo.py
│       ├── elconfidencial.py
│       ├── eldiario.py
│       ├── publico.py
│       └── okdiario.py
│
├── ground_truth/                  # Etiquetado automático con LLMs
│   ├── pipeline.py                # Orquestador principal
│   ├── llm_clients.py             # Clientes Claude / GPT / Llama (Groq)
│   ├── voting.py                  # Sistema de votación ponderada
│   ├── data.py                    # Carga, limpieza y muestreo
│   └── rerun_*.py                 # Reintentos por proveedor
│
├── notebooks/
│   ├── 1_limpieza.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_embedings.ipynb
│   ├── 4_clustering.ipynb
│   ├── 5.0_ground_truth.ipynb
│   ├── 5.1_supervisado_semisupervisado.ipynb
│   ├── 5.2_finetuning_colab/      # 8 variantes LoRA + comparativa
│   ├── 5.2_zeroshot_fewshot.ipynb
│   ├── 5.3_comparativa_test_modelos.ipynb
│   ├── 6_inferencia_colab.ipynb
│   ├── 7.0_dataframe_final.ipynb
│   └── 7.1_reranking.ipynb
│
├── data/
│   ├── processed/                 # corpus_limpio_filtrado.parquet (241 240 art.)
│   ├── ground_truth/              # Etiquetas LLM y ground truth limpio
│   ├── splits/                    # Parquet y .npy de train/val/test/unlabeled
│   └── final/                     # corpus_maestro.parquet (resultado final)
│
├── embeddings/                    # Vectores .npy por medio (Qwen3-0.6B, 1 024d)
├── checkpoints/                   # Estado del scraper por medio
├── output/
│   ├── clustering_6medios/        # Modelos BERTopic y visualizaciones
│   ├── clasificacion_supervisada/ # Modelos locales y métricas
│   ├── finetuning_predictions/    # Predicciones por artículo de cada variante LoRA
│   ├── reranking/                 # Visualizaciones del reranking
│   ├── finetuning_test_metrics.csv
│   ├── finetuning_val_metrics.csv
│   ├── zeroshot_results.csv
│   ├── fewshot_results.csv
│   └── inferencia_ideologica.csv
└── logs/
```

---

## Instalación

```bash
git clone <repo>
cd TFM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para los notebooks de clasificación y fine-tuning, instalar también:

```bash
pip install torch transformers peft datasets accelerate
pip install bertopic umap-learn optuna sentence-transformers
pip install anthropic openai groq
pip install pandas numpy scikit-learn plotly
```

Copiar la plantilla de variables de entorno y rellenar las claves:

```bash
cp .env.example .env
```

Editar `.env` con las claves reales:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here   # opcional
```

> El fine-tuning (notebooks `5.2_finetuning_colab/`) requiere GPU y está diseñado para ejecutarse en Google Colab con acceso a Drive.

---

## Resultados principales

### Comparativa de modelos (F1-macro, test set, n=238)

| Modelo | F1-macro val | F1-macro test | Δ val→test |
|---|---|---|---|
| **LoRA QKV r8** (ganador) | 0.821 | **0.830** | **+0.009** |
| LoRA All-linear r8 | 0.829 | 0.823 | −0.006 |
| LoRA QV r8 | 0.818 | 0.816 | −0.002 |
| LoRA Full-7layers r8 | 0.803 | 0.810 | +0.007 |
| LoRA Regularized r8 | 0.822 | 0.806 | −0.016 |
| LoRA MLP-head r32 | 0.800 | 0.795 | −0.005 |
| QLoRA QKV r8 | 0.791 | 0.788 | −0.003 |
| MLP supervisado | — | ~0.62 | — |
| TF-IDF + SVM | — | ~0.58 | — |

### Corpus final

El fichero `data/final/corpus_maestro.parquet` contiene los 241 240 artículos con:

- `label` — clase ideológica (IZQUIERDA / CENTRO / DERECHA)
- `label_source` — `gold` (ground truth manual/LLM) o `inferred` (modelo LoRA)
- `p_izquierda`, `p_centro`, `p_derecha` — probabilidades del modelo
- `topic` — macrotema asignado por BERTopic
- Metadatos originales: `medio`, `fecha`, `tema`, `titular`, `abstract`, `noticia`, `url`

---

## Tecnologías

| Área | Herramientas |
|---|---|
| Scraping | `requests`, `BeautifulSoup`, `feedparser` |
| Procesamiento | `pandas`, `numpy`, `spaCy` |
| Embeddings | `Qwen/Qwen3-0.6B`, `sentence-transformers` |
| Reducción dimensional | `UMAP` |
| Clustering | `BERTopic`, `DBSCAN`, `Optuna` |
| Modelos clásicos | `scikit-learn` (TF-IDF, LR, SVM, LP, LS) |
| Deep learning | `PyTorch`, `Transformers`, `PEFT` |
| Fine-tuning | `Qwen/Qwen3-4B` + LoRA/QLoRA (Colab A100) |
| LLMs árbitros | Claude Sonnet 4.6, GPT-5.2, Llama 3.1 8B (Groq) |
| Visualización | `plotly`, `matplotlib`, `seaborn` |

---