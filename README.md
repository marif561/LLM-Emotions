<div align="center">

# Exploring Multidimensional Aspects of Hope Speech Computationally
### A Psycholinguistic and Emotional Perspective

**Muhammad Arif · Moein Shahiki Tash · Ainaz Jamshidi · Iqra Ameer · Fida Ullah · Jugal Kalita · Alexander Gelbukh · Fazlourrahman Balouchzahi**

Instituto Politécnico Nacional (CIC-IPN) · University of Maryland Baltimore County · Penn State Abington · University of Colorado Colorado Springs

[![Project page](https://img.shields.io/badge/Project-Page-d97706)](https://marif561.github.io/LLM-Emotions/)
![Status](https://img.shields.io/badge/Status-Submitted%20to%20Scientific%20Reports-db2777)
![Dataset](https://img.shields.io/badge/Dataset-PolyHope-8b5cf6)
![Best binary macro-F1](https://img.shields.io/badge/Binary%20macro--F1-0.8161-10b981)

</div>

---

## Overview

Hope is a vital coping mechanism — but it is not always positive. People hope for things out of joy, and also out of anger, sadness or despair. This project analyses social-media posts with three NLP tools to extract **psycholinguistic** (LIWC), **emotional** (NRC Emotion Lexicon) and **sentiment** (VADER) features, shows that different types of hope carry distinct emotional and cognitive signatures, and uses these features to classify hope with gradient-boosting models.

**Highlights**

- Different hope types have measurably different emotional profiles — Generalized Hope peaks in joy, anticipation and trust; Unrealistic Hope peaks in anger, sadness and negative sentiment.
- Adding psycholinguistic, emotional and sentiment features beats n-grams alone for every model.
- **LightGBM** with all features reaches **0.8161 macro-F1** (binary) and **0.5800 macro-F1** (multiclass) — about **4 points above** the traditional TF-IDF machine-learning baseline on multiclass, at a fraction of the compute of deep models.

## The task: four kinds of hope

| Category | Definition | Tweets |
|---|---|---|
| **Not Hope** | No expressed hope, desire or expectation for the future | 4,081 |
| **Generalized Hope** | Broad optimism not tied to a specific event or outcome | 2,335 |
| **Realistic Hope** | Anticipating outcomes that are sensible, significant and achievable | 982 |
| **Unrealistic Hope** | Wishes that defy rational expectation, often driven by anger, sadness or despair | 858 |
| | **Total** | **8,256** |

Data: the **PolyHope** English Twitter dataset (Balouchzahi et al., 2023). Classification is done at two levels — **binary** (Hope vs. Not Hope) and **multiclass** (all four categories).

## Methodology

```
Tweets ──► Preprocess ──► Feature extraction ──► Feature selection ──► Bayesian tuning ──► 5-fold CV
           emoji→text      LIWC-22                SelectFromModel        LightGBM
           abbreviations   NRC emotions (8)       (random forest,        CatBoost
           stopwords       VADER sentiment (4)    n-grams only)          XGBoost
                           char + word n-grams
```

| Feature type | LIWC | Emotions | Sentiments | Char n-grams | Word n-grams |
|---|---|---|---|---|---|
| All | 8,250 | 8 | 4 | 361 | 23,438 |
| After selection | — | — | — | 37 | 4,097 |

Each classifier is trained with three feature sets — **n-grams only**, **LIWC + emotions + sentiments**, and **all features combined** — using both default and Bayesian-optimized hyperparameters.

## Emotional profile of each hope type

Average NRC emotion scores (higher = stronger association):

| Emotion | Not Hope | Generalized | Realistic | Unrealistic |
|---|---|---|---|---|
| Anger | 0.55 | 0.35 | 0.43 | **0.56** |
| Anticipation | 1.21 | **1.60** | 1.32 | 0.90 |
| Disgust | **0.40** | 0.25 | 0.32 | **0.40** |
| Fear | **0.86** | 0.59 | 0.65 | 0.71 |
| Joy | 0.89 | **1.61** | 1.17 | 0.84 |
| Sadness | 0.57 | 0.43 | 0.42 | **0.61** |
| Surprise | 0.63 | **0.95** | 0.72 | 0.49 |
| Trust | 1.46 | **1.61** | 1.43 | 0.87 |

VADER sentiment: positive sentiment is strongest in Generalized (0.37) and Realistic (0.31) Hope, while negative sentiment is highest in Unrealistic Hope (0.14).

## Results

Macro-F1, 5-fold cross-validation, **tuned** hyperparameters:

| Model | N-grams | LIWC + Emotions + Sentiments | All features |
|---|---|---|---|
| **Binary** | | | |
| LightGBM | 0.7974 | 0.8027 | **0.8161** |
| CatBoost | 0.8003 | 0.8154 | 0.8014 |
| XGBoost | 0.7870 | 0.7923 | 0.8068 |
| **Multiclass** | | | |
| LightGBM | 0.5368 | 0.5507 | **0.5800** |
| CatBoost | 0.5350 | 0.5414 | 0.5539 |
| XGBoost | 0.5308 | 0.5412 | 0.5745 |

### Comparison with PolyHope baselines

| Model | Approach | Weighted F1 | Macro F1 |
|---|---|---|---|
| *Binary* | | | |
| LR (Balouchzahi et al., 2023) | Machine learning | 0.7979 | 0.7978 |
| **CatBoost (proposed features)** | Machine learning | **0.8155** | **0.8154** |
| BiLSTM / CNN + GloVe | Deep learning | 0.8230 | 0.8229 |
| BERT, RoBERTa, XLNet | Transformers | 0.8523 | 0.8522 |
| *Multiclass* | | | |
| CatBoost (Balouchzahi et al., 2023) | Machine learning | 0.6416 | 0.5350 |
| **LightGBM (proposed features)** | Machine learning | **0.6793** | **0.5800** |
| CNN + GloVe | Deep learning | 0.6962 | 0.6135 |
| BERT | Transformers | 0.7701 | 0.7193 |

Feature-based gradient boosting beats the traditional TF-IDF baseline and comes close to deep learning with far fewer resources; transformers remain ahead.

## Key findings

- **Generalized Hope** — tied to joy, optimism and enthusiasm; more affect-laden language; less focused on practical, everyday matters.
- **Realistic Hope** — grounded in what is achievable; strongest "drives"; least conversational, suggesting a reserved, pragmatic tone; linked to work, home, school and cultural or political topics.
- **Unrealistic Hope** — fewer words but higher cognitive engagement; more anger, sadness and negative sentiment; more common in discussions of health.

## Repository contents

| File | Description |
|---|---|
| `README.md` | This file |
| `index.html` | Interactive project page (served via GitHub Pages) |
| `CITATION.cff` | Citation metadata — enables GitHub's "Cite this repository" button |
| `.gitignore` | Keeps caches, checkpoints and secrets out of the repo |

Feature-extraction scripts and analysis notebooks will be added here.

## Data availability

The PolyHope dataset is described in Balouchzahi et al. (2023). The data used in this study are available from the corresponding author on reasonable request.

## Citation

```bibtex
@misc{arif_hope_speech,
  title  = {Exploring Multidimensional Aspects of Hope Speech Computationally:
            A Psycholinguistic and Emotional Perspective},
  author = {Arif, Muhammad and Shahiki Tash, Moein and Jamshidi, Ainaz and
            Ameer, Iqra and Ullah, Fida and Kalita, Jugal and
            Gelbukh, Alexander and Balouchzahi, Fazlourrahman},
  note   = {Preprint submitted to Scientific Reports}
}
```

## Contact

Muhammad Arif — [GitHub @marif561](https://github.com/marif561) · [LinkedIn](https://www.linkedin.com/in/muhammadarif1906/)
Corresponding author: F. Balouchzahi (fbalouchzahi2021@cic.ipn.mx)
