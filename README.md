# 📊 Data Science Studio

> Plataforma completa de Machine Learning implementada em **Python puro** — sem NumPy, sem Pandas, sem scikit-learn. Todos os algoritmos foram construídos do zero, incluindo álgebra linear, estatística e ML. Interface web com Flask.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat&logo=flask)
![ML](https://img.shields.io/badge/ML-from_scratch-FF6B35?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Demonstração

```
pip install flask
python main.py
# Abra: http://localhost:5000
```

---

## 🧠 Algoritmos Implementados

Todos os algoritmos foram escritos **do zero em Python puro**, sem nenhuma biblioteca de ML externa.

### Machine Learning

| Algoritmo | Técnica | Validado |
|-----------|---------|----------|
| **K-Means++** | Clustering com inicialização otimizada, Elbow + Silhouette | `sil=0.71` |
| **Regressão OLS** | Mínimos Quadrados via eliminação de Gauss-Jordan, grau 1–5 | `R²=0.99` |
| **Naive Bayes Gaussiano** | Probabilidade condicional com log-likelihood | `acc=100%` |
| **PCA** | Power Iteration na matriz de covariância | `var=95%` |
| **Árvore de Decisão ID3** | Entropia + Information Gain recursivo | `acc=97%` |
| **KNN** | K-Nearest Neighbors com k-sweep automático | `acc=100%` |

### Análise de Dados

- **Estatísticas Descritivas** — mean, std, quartis, skewness, kurtosis
- **Correlação de Pearson** — matriz completa com heatmap
- **Série Temporal** — tendência linear, média móvel, ACF, detecção de anomalias (Z-score)
- **Normalização** — Z-Score e Min-Max

### Álgebra Linear (from scratch)

```python
dot()         # produto escalar
mat_mul()     # multiplicação de matrizes
transpose()   # transposição
cov_matrix()  # matriz de covariância
zscore_scale()# normalização Z-Score
```

---

## 🗂️ Estrutura do Projeto

```
data-science-studio/
│
├── main.py              # Arquivo único — backend + frontend
│   │
│   ├── [MÓDULO 1] Álgebra Linear
│   │   └── dot, mat_mul, transpose, cov_matrix, ...
│   │
│   ├── [MÓDULO 2] Estatísticas
│   │   └── describe(), correlation_matrix()
│   │
│   ├── [MÓDULO 3] Pré-processamento
│   │   └── zscore_scale(), minmax_scale(), polynomial_features()
│   │
│   ├── [MÓDULO 4] K-Means++
│   │   └── kmeans() — inicialização K-Means++, Elbow, Silhouette
│   │
│   ├── [MÓDULO 5] Regressão OLS
│   │   └── linear_regression() — Gauss-Jordan, bandas de confiança
│   │
│   ├── [MÓDULO 6] Naive Bayes
│   │   └── class GaussianNaiveBayes — fit, predict, predict_proba
│   │
│   ├── [MÓDULO 7] PCA
│   │   └── pca() — Power Iteration, variância explicada
│   │
│   ├── [MÓDULO 8] Árvore de Decisão
│   │   └── build_tree(), predict_tree() — ID3, entropia, info_gain
│   │
│   ├── [MÓDULO 9] KNN
│   │   └── knn_predict() — k-sweep com z-score scaling
│   │
│   ├── [MÓDULO 10] Datasets Sintéticos
│   │   └── gen_blobs, gen_moons, gen_sales, gen_iris_like, gen_timeseries
│   │
│   ├── [MÓDULO 11] Flask API (15 endpoints)
│   │   └── /api/ml/*, /api/dataset/*, /api/timeseries/*
│   │
│   └── [MÓDULO 12] Interface Web
│       └── HTML + CSS + JS inline (SVG charts, dark theme premium)
│
├── requirements.txt     # Apenas: flask
└── README.md
```

---

## 🌐 API REST

### Dataset

```http
POST /api/dataset/generate
Content-Type: application/json

{ "name": "iris", "n": 150, "k": 3 }
```

Datasets disponíveis: `blobs`, `moons`, `regression`, `sales`, `iris`, `timeseries`

```http
POST /api/dataset/upload        # Upload de CSV raw
GET  /api/dataset/stats         # Estatísticas descritivas + correlação
GET  /api/dataset/full          # Dataset completo (max 1000 rows)
```

### Machine Learning

```http
POST /api/ml/kmeans
{ "features": ["x", "y"], "k": 3, "normalize": true }

POST /api/ml/regression
{ "x_col": "temperatura", "y_col": "vendas", "degree": 2 }

POST /api/ml/naivebayes
{ "features": ["f1", "f2"], "target": "label" }

POST /api/ml/pca
{ "features": ["f1", "f2", "f3"], "n_components": 2 }

POST /api/ml/tree
{ "features": ["f1", "f2"], "target": "label", "max_depth": 4 }

POST /api/ml/knn
{ "features": ["f1", "f2"], "target": "label", "k": 5 }
```

### Análise Temporal

```http
POST /api/timeseries/analyze
{ "col": "value" }
```

### Exemplo de resposta — K-Means

```json
{
  "labels": [0, 1, 2, 0, 1, ...],
  "centroids": [[1.2, -0.4], [0.1, 1.8], [-1.5, 0.3]],
  "inertia": 42.49,
  "silhouette": 0.71,
  "iterations": 12,
  "cluster_sizes": {"0": 73, "1": 75, "2": 76},
  "elbow": [{"k": 2, "inertia": 367.8, "silhouette": 0.363}, ...]
}
```

---

## 🎨 Interface

- **Dark theme premium** com fundo `#080c14`
- **Fontes** — Rajdhani (UI) + Fira Code (dados)
- **Gráficos SVG puros** — scatter, line chart, bar chart sem biblioteca externa
- **Sidebar colapsável** com logo hexagonal animado
- **Integração com Claude LLM** — análise generativa dos resultados
- **10 páginas** — Dashboard, Dataset, Estatísticas, K-Means, Regressão, Classificação, PCA, Série Temporal, LLM, Histórico

---

## 🚀 Como Rodar

### Requisitos

- Python 3.8+
- Flask

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/data-science-studio.git
cd data-science-studio

# 2. Instale a dependência
pip install flask

# 3. Rode o servidor
python main.py

# 4. Abra no navegador
# http://localhost:5000
```

### Windows (PowerShell)

```powershell
pip install flask
python main.py
# Abra: http://localhost:5000
```

---

## 📦 Dependências

```
flask>=2.0
```

**Isso é tudo.** Toda a matemática de ML — álgebra linear, estatística, otimização — foi implementada usando apenas a **biblioteca padrão do Python** (`math`, `random`, `statistics`, `collections`, `csv`, `io`).

---

## 🧪 Resultados Validados

```
K-Means++   → Silhouette = 0.71   (blobs k=4, n=300)
Regressão   → R² = 0.99           (dados lineares)
Naive Bayes → Acurácia = 100%     (iris-like, 36 amostras de teste)
PCA         → Variância = 95.26%  (2 componentes de 4 features)
Árvore ID3  → Acurácia = 97.2%    (iris-like, depth=4)
KNN         → Acurácia = 100%     (k=3,5,7,9 — iris-like)
```

---

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/novo-algoritmo`
3. Commit: `git commit -m "feat: adiciona Random Forest from scratch"`
4. Push: `git push origin feature/novo-algoritmo`
5. Abra um Pull Request

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE) para detalhes.

---

<p align="center">Feito com Python puro 🐍 — zero dependências de ML</p>
