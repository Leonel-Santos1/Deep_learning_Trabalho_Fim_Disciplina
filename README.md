# Classificação de Gênero Musical com CNN sobre Mel Spectrograms

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r79GKQ0EUb8yX8ptK_m6vP0Wu70iHRay?usp=sharing)

## Identificação

- **Nome do estudante:** Francisco Leonel Ferreira dos Santos
- **Disciplina:** Aprendizado Profundo
- **Modelo implementado:** CNN VGG-like (4 blocos convolucionais com BatchNorm, ReLU, MaxPooling e Dropout)
- **Dataset utilizado:** GTZAN Music Genre Dataset
- **Link do dataset:** https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## Descrição do Problema

Classificação automática de gênero musical a partir de clipes de áudio.
Cada clipe é convertido em um Mel Spectrogram — uma representação visual da
distribuição de energia sonora ao longo do tempo em escala logarítmica —
transformando o problema de áudio em um problema de visão computacional.
A tarefa é de **classificação multiclasse**: atribuir cada clipe a um dos
10 gêneros musicais presentes no dataset.

## Dataset

| Característica       | Valor                                                                 |
|----------------------|-----------------------------------------------------------------------|
| Total de amostras    | 1.000 clipes (999 válidos — 1 arquivo de jazz corrompido descartado) |
| Classes              | 10 gêneros: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| Amostras por classe  | 100 (dataset balanceado)                                             |
| Formato original     | WAV, 22.050 Hz, mono, 30 segundos por clipe                         |
| Formato de entrada   | Imagens PNG 128×128 px (Mel Spectrograms normalizados para [0, 1])  |
| Treino               | 699 amostras (70%)                                                   |
| Validação            | 150 amostras (15%)                                                   |
| Teste                | 150 amostras (15%)                                                   |
| Estratificação       | Por classe, com seed 42                                              |

## Modelo

Arquitetura CNN VGG-like com 4 blocos convolucionais de profundidade crescente:

```
Entrada (3×128×128)
  → Bloco 1: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.2)
  → Bloco 2: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.3)
  → Bloco 3: Conv2D(256) → BN → ReLU → Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.4)
  → Bloco 4: Conv2D(512) → BN → ReLU → Conv2D(512) → BN → ReLU → MaxPool
  → Global Average Pooling → FC(256) → ReLU → Dropout(0.5) → FC(10) → Softmax
```

**Hiperparâmetros:**

| Parâmetro                  | Valor             |
|----------------------------|-------------------|
| Otimizador                 | SGD               |
| Taxa de aprendizado        | 0,025             |
| Momentum                   | 0,9               |
| Weight decay               | 5×10⁻⁴            |
| Scheduler                  | CosineAnnealingLR |
| Épocas máximas             | 150               |
| Early stopping (patience)  | 60                |
| Batch size                 | 64                |
| Função de perda            | CrossEntropyLoss  |
| Label smoothing            | 0,1               |
| Seed                       | 42                |

## Ambiente

- **Linguagem:** Python 3.10
- **Bibliotecas:** torch, librosa, numpy, Pillow, scikit-learn, matplotlib, seaborn, tqdm, kagglehub
- **Uso de GPU:** Sim (NVIDIA T4 — Google Colab)

## Principais Resultados

Duas execuções independentes com os mesmos hiperparâmetros foram realizadas,
com variação atribuída à estocasticidade do SGD:

| Execução | Melhor época | Acurácia no teste |
|----------|-------------|-------------------|
| 1        | 129         | 83%               |
| 2        | 75          | 81%               |

**Métricas por classe (execução com 81%):**

| Gênero    | Precisão | Recall | F1   |
|-----------|----------|--------|------|
| blues     | 0,86     | 0,80   | 0,83 |
| classical | 0,93     | 0,93   | 0,93 |
| country   | 0,74     | 0,93   | 0,82 |
| disco     | 0,82     | 0,60   | 0,69 |
| hiphop    | 0,80     | 0,80   | 0,80 |
| jazz      | 0,88     | 1,00   | 0,94 |
| metal     | 0,78     | 0,93   | 0,85 |
| pop       | 0,68     | 0,87   | 0,76 |
| reggae    | 0,87     | 0,87   | 0,87 |
| rock      | 0,71     | 0,33   | 0,45 |
| **Média** | **0,81** | **0,81** | **0,79** |

O modelo também foi testado com uma música real (*Awake and Alive* — Skillet),
classificada corretamente como **metal** com 70,5% de confiança.

## Como Executar

O projeto foi desenvolvido inteiramente em **Google Colab**. Para reproduzir:

### 1. Abrir o notebook no Colab

Faça upload do arquivo `CNN_GTZAN.ipynb` no Google Colab.

### 2. Ativar GPU

`Runtime → Change runtime type → T4 GPU`

### 3. Executar as células em ordem

O notebook está organizado nas seguintes seções:

| Seção | Descrição |
|-------|-----------|
| 0. Seed | Configuração de reprodutibilidade |
| 1. Download | Download automático do dataset GTZAN via `kagglehub` (sem necessidade de login) |
| 2. Análise Exploratória | Visualização de formas de onda e Mel Spectrograms |
| 3. Pré-processamento | Conversão dos WAVs em imagens PNG |
| 4. Carregamento | Leitura das imagens e divisão treino/val/teste |
| 5. Arquitetura CNN | Definição do modelo |
| 6. DataLoaders | Preparação dos batches |
| 7. Treinamento | Loop de treino com early stopping |
| 8. Avaliação | Métricas, matriz de confusão e exemplos visuais |
| 9. Predição | Teste com música real (upload de arquivo `.mp3` ou `.wav`) |
