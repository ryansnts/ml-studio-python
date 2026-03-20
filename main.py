"""
╔══════════════════════════════════════════════════════════════╗
║         DATA SCIENCE STUDIO — Python ML Platform            ║
║  Algoritmos: KMeans · Regressão · Naive Bayes · PCA         ║
║  Features: Dataset Upload · Visualização · Predição · LLM   ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── stdlib only (+ flask) ─────────────────────────────────────
import math, random, json, re, csv, io, statistics, time, os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
random.seed(42)


# ══════════════════════════════════════════════════════════════
#  NÚCLEO DE ÁLGEBRA LINEAR (sem numpy)
# ══════════════════════════════════════════════════════════════

def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def vec_add(a, b):
    return [x+y for x,y in zip(a,b)]

def vec_sub(a, b):
    return [x-y for x,y in zip(a,b)]

def vec_scale(a, s):
    return [x*s for x in a]

def vec_norm(a):
    return math.sqrt(sum(x*x for x in a)) or 1e-10

def vec_dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def mat_mul(A, B):
    rows, cols = len(A), len(B[0])
    k = len(B)
    return [[sum(A[i][p]*B[p][j] for p in range(k)) for j in range(cols)] for i in range(rows)]

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def mat_vec(M, v):
    return [dot(row, v) for row in M]

def outer(a, b):
    return [[x*y for y in b] for x in a]

def mean_vec(data):
    n = len(data)
    d = len(data[0])
    return [sum(data[i][j] for i in range(n))/n for j in range(d)]

def cov_matrix(data):
    mu = mean_vec(data)
    n, d = len(data), len(data[0])
    C = [[0.0]*d for _ in range(d)]
    for x in data:
        diff = vec_sub(x, mu)
        for i in range(d):
            for j in range(d):
                C[i][j] += diff[i]*diff[j]
    return [[C[i][j]/(n-1) for j in range(d)] for i in range(d)]


# ══════════════════════════════════════════════════════════════
#  ESTATÍSTICAS DESCRITIVAS
# ══════════════════════════════════════════════════════════════

def describe(col: List[float]) -> Dict:
    col = [x for x in col if x is not None]
    if not col:
        return {}
    n = len(col)
    s = sorted(col)
    mu = sum(col)/n
    variance = sum((x-mu)**2 for x in col)/(n-1) if n > 1 else 0
    std = math.sqrt(variance)
    q1 = s[n//4]
    median = s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
    q3 = s[3*n//4]
    skew = sum((x-mu)**3 for x in col)/(n*std**3) if std else 0
    kurt = sum((x-mu)**4 for x in col)/(n*std**4) - 3 if std else 0
    return {
        "n": n, "mean": round(mu,4), "std": round(std,4),
        "min": round(s[0],4), "q1": round(q1,4), "median": round(median,4),
        "q3": round(q3,4), "max": round(s[-1],4),
        "skew": round(skew,4), "kurt": round(kurt,4),
        "range": round(s[-1]-s[0],4), "iqr": round(q3-q1,4)
    }

def correlation_matrix(cols: Dict[str, List[float]]) -> Dict:
    names = list(cols.keys())
    n = len(names)
    matrix = {}
    for i in range(n):
        for j in range(n):
            a, b = cols[names[i]], cols[names[j]]
            pairs = [(x,y) for x,y in zip(a,b) if x is not None and y is not None]
            if len(pairs) < 2:
                matrix[f"{names[i]}|{names[j]}"] = 0.0
                continue
            xs, ys = zip(*pairs)
            mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
            num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
            denom = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
            matrix[f"{names[i]}|{names[j]}"] = round(num/denom,4) if denom else 0.0
    return {"names": names, "matrix": matrix}


# ══════════════════════════════════════════════════════════════
#  NORMALIZAÇÃO / PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════

def minmax_scale(data: List[List[float]]) -> Tuple[List[List[float]], List, List]:
    if not data: return [], [], []
    d = len(data[0])
    mins = [min(row[j] for row in data) for j in range(d)]
    maxs = [max(row[j] for row in data) for j in range(d)]
    scaled = []
    for row in data:
        scaled.append([(row[j]-mins[j])/(maxs[j]-mins[j]) if maxs[j]!=mins[j] else 0.0
                       for j in range(d)])
    return scaled, mins, maxs

def zscore_scale(data: List[List[float]]) -> Tuple[List[List[float]], List, List]:
    if not data: return [], [], []
    d = len(data[0])
    means = [sum(row[j] for row in data)/len(data) for j in range(d)]
    stds  = [math.sqrt(sum((row[j]-means[j])**2 for row in data)/len(data)) or 1.0
             for j in range(d)]
    scaled = [[(row[j]-means[j])/stds[j] for j in range(d)] for row in data]
    return scaled, means, stds


# ══════════════════════════════════════════════════════════════
#  ALGORITMO 1 — K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════

def kmeans(data: List[List[float]], k: int = 3, max_iter: int = 100) -> Dict:
    if not data or k <= 0: return {}
    n, d = len(data), len(data[0])
    k = min(k, n)

    # Inicialização K-Means++
    centroids = [random.choice(data)[:]]
    for _ in range(k-1):
        dists = [min(vec_dist(x, c)**2 for c in centroids) for x in data]
        total = sum(dists)
        r = random.uniform(0, total)
        cum = 0
        for i, d2 in enumerate(dists):
            cum += d2
            if cum >= r:
                centroids.append(data[i][:])
                break
        else:
            centroids.append(random.choice(data)[:])

    labels = [0] * n
    history = []

    for iteration in range(max_iter):
        # Assign
        new_labels = []
        for x in data:
            dists = [vec_dist(x, c) for c in centroids]
            new_labels.append(dists.index(min(dists)))

        # Inertia
        inertia = sum(vec_dist(data[i], centroids[new_labels[i]])**2 for i in range(n))
        history.append(round(inertia, 4))

        if new_labels == labels and iteration > 0:
            break
        labels = new_labels

        # Update centroids
        new_centroids = []
        for ki in range(k):
            members = [data[i] for i in range(n) if labels[i] == ki]
            if members:
                new_centroids.append(mean_vec(members))
            else:
                new_centroids.append(random.choice(data)[:])
        centroids = new_centroids

    # Silhouette score (simplified)
    def silhouette_sample(i):
        same = [data[j] for j in range(n) if labels[j]==labels[i] and j!=i]
        a = sum(vec_dist(data[i],x) for x in same)/len(same) if same else 0
        other_scores = []
        for ki in range(k):
            if ki != labels[i]:
                others = [data[j] for j in range(n) if labels[j]==ki]
                if others:
                    other_scores.append(sum(vec_dist(data[i],x) for x in others)/len(others))
        b = min(other_scores) if other_scores else 0
        return (b-a)/max(a,b) if max(a,b) > 0 else 0

    sample_idx = random.sample(range(n), min(50, n))
    silhouette = sum(silhouette_sample(i) for i in sample_idx)/len(sample_idx)

    sizes = Counter(labels)
    return {
        "labels": labels,
        "centroids": [[round(v,4) for v in c] for c in centroids],
        "inertia": round(history[-1], 4),
        "inertia_history": history,
        "silhouette": round(silhouette, 4),
        "cluster_sizes": {str(k): v for k, v in sizes.items()},
        "iterations": len(history)
    }


# ══════════════════════════════════════════════════════════════
#  ALGORITMO 2 — REGRESSÃO LINEAR / POLINOMIAL (OLS)
# ══════════════════════════════════════════════════════════════

def linear_regression(X: List[List[float]], y: List[float]) -> Dict:
    n = len(X)
    if n < 2: return {}
    d = len(X[0])

    # Adiciona bias (coluna de 1s)
    Xb = [[1.0] + row for row in X]
    Xt = transpose(Xb)
    # β = (XᵀX)⁻¹ Xᵀy   — via pseudo-inversa iterativa (gradiente)
    # Para d pequeno, usamos solução fechada via eliminação de Gauss

    # Xᵀ X
    XtX = mat_mul(Xt, Xb)
    # Xᵀ y
    Xty = [dot(row, y) for row in Xt]

    # Gauss-Jordan para resolver XtX @ beta = Xty
    size = len(XtX)
    aug = [XtX[i][:] + [Xty[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            continue
        aug[col] = [v/aug[col][col] for v in aug[col]]
        for row in range(size):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor*aug[col][j] for j in range(size+1)]

    beta = [aug[i][-1] for i in range(size)]
    intercept = beta[0]
    coefficients = beta[1:]

    # Predictions & metrics
    y_pred = [intercept + dot(coefficients, x) for x in X]
    y_mean = sum(y)/n
    ss_res = sum((y[i]-y_pred[i])**2 for i in range(n))
    ss_tot = sum((y[i]-y_mean)**2 for i in range(n))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    rmse = math.sqrt(ss_res/n)
    mae  = sum(abs(y[i]-y_pred[i]) for i in range(n))/n

    # Residuals
    residuals = [round(y[i]-y_pred[i], 4) for i in range(n)]

    return {
        "intercept": round(intercept, 6),
        "coefficients": [round(c, 6) for c in coefficients],
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "y_pred": [round(v, 4) for v in y_pred],
        "residuals": residuals,
        "n_samples": n,
        "n_features": d
    }


def polynomial_features(X_1d: List[float], degree: int = 2) -> List[List[float]]:
    return [[x**p for p in range(1, degree+1)] for x in X_1d]


# ══════════════════════════════════════════════════════════════
#  ALGORITMO 3 — NAIVE BAYES GAUSSIANO
# ══════════════════════════════════════════════════════════════

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X: List[List[float]], y: List):
        self.classes = list(set(y))
        n = len(y)
        for c in self.classes:
            idx = [i for i,yi in enumerate(y) if yi==c]
            self.priors[c] = len(idx)/n
            Xc = [X[i] for i in idx]
            self.means[c] = mean_vec(Xc)
            d = len(X[0])
            self.vars[c] = [
                sum((Xc[j][f]-self.means[c][f])**2 for j in range(len(Xc)))/len(Xc) + 1e-9
                for f in range(d)
            ]
        return self

    def _log_likelihood(self, x, c):
        ll = math.log(self.priors[c])
        for f, (mu, var) in enumerate(zip(self.means[c], self.vars[c])):
            ll -= 0.5*math.log(2*math.pi*var)
            ll -= (x[f]-mu)**2/(2*var)
        return ll

    def predict_proba(self, X: List[List[float]]) -> List[Dict]:
        results = []
        for x in X:
            log_probs = {c: self._log_likelihood(x, c) for c in self.classes}
            max_lp = max(log_probs.values())
            probs = {c: math.exp(lp - max_lp) for c, lp in log_probs.items()}
            total = sum(probs.values())
            probs = {c: round(v/total, 4) for c, v in probs.items()}
            results.append(probs)
        return results

    def predict(self, X):
        return [max(p, key=p.get) for p in self.predict_proba(X)]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for p,t in zip(preds,y) if p==t)/len(y)


# ══════════════════════════════════════════════════════════════
#  ALGORITMO 4 — PCA (Power Iteration)
# ══════════════════════════════════════════════════════════════

def pca(data: List[List[float]], n_components: int = 2) -> Dict:
    n = len(data)
    if n < 2: return {}
    d = len(data[0])
    n_components = min(n_components, d, n)

    mu = mean_vec(data)
    centered = [vec_sub(row, mu) for row in data]
    C = cov_matrix(centered)

    # Power iteration para encontrar eigenvectors
    eigenvectors = []
    C_deflated = [row[:] for row in C]

    for _ in range(n_components):
        v = [random.gauss(0,1) for _ in range(d)]
        norm = vec_norm(v)
        v = [x/norm for x in v]

        for _iter in range(200):
            v_new = mat_vec(C_deflated, v)
            norm = vec_norm(v_new)
            if norm < 1e-10: break
            v_new = [x/norm for x in v_new]
            if vec_dist(v_new, v) < 1e-8: break
            v = v_new

        eigenvalue = dot(mat_vec(C_deflated, v), v)

        # Deflate
        outer_vv = outer(v, v)
        C_deflated = [[C_deflated[i][j] - eigenvalue*outer_vv[i][j]
                       for j in range(d)] for i in range(d)]
        eigenvectors.append((eigenvalue, v))

    # Explained variance
    total_var = sum(abs(ev) for ev, _ in eigenvectors) or 1.0
    explained = [round(abs(ev)/total_var, 4) for ev, _ in eigenvectors]

    # Project data
    components = [vec for _, vec in eigenvectors]
    projected = [[round(dot(row, comp), 4) for comp in components] for row in centered]

    return {
        "projected": projected,
        "components": [[round(v,4) for v in comp] for comp in components],
        "explained_variance_ratio": explained,
        "cumulative_variance": [round(sum(explained[:i+1]),4) for i in range(len(explained))],
        "n_components": n_components
    }


# ══════════════════════════════════════════════════════════════
#  ALGORITMO 5 — ÁRVORE DE DECISÃO (ID3 simplificado)
# ══════════════════════════════════════════════════════════════

def entropy(y):
    if not y: return 0
    counts = Counter(y)
    n = len(y)
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c > 0)

def best_split(X, y, feature_names):
    best_gain, best_feat, best_thresh = 0, None, None
    base_ent = entropy(y)
    n = len(y)
    for fi in range(len(X[0])):
        vals = sorted(set(x[fi] for x in X))
        thresholds = [(vals[i]+vals[i+1])/2 for i in range(len(vals)-1)]
        for t in thresholds[:20]:  # limit for speed
            left_y  = [y[i] for i in range(n) if X[i][fi] <= t]
            right_y = [y[i] for i in range(n) if X[i][fi] >  t]
            if not left_y or not right_y: continue
            gain = base_ent - (len(left_y)/n)*entropy(left_y) - (len(right_y)/n)*entropy(right_y)
            if gain > best_gain:
                best_gain = gain
                best_feat = fi
                best_thresh = t
    return best_feat, best_thresh, round(best_gain, 4)

def build_tree(X, y, feature_names, depth=0, max_depth=4):
    if not y or len(set(y))==1 or depth>=max_depth or len(y)<5:
        label = Counter(y).most_common(1)[0][0] if y else None
        return {"type":"leaf","label":label,"samples":len(y),"distribution":dict(Counter(y))}
    fi, thresh, gain = best_split(X, y, feature_names)
    if fi is None:
        label = Counter(y).most_common(1)[0][0]
        return {"type":"leaf","label":label,"samples":len(y),"distribution":dict(Counter(y))}
    n = len(y)
    left_mask  = [i for i in range(n) if X[i][fi] <= thresh]
    right_mask = [i for i in range(n) if X[i][fi] >  thresh]
    return {
        "type": "node",
        "feature": feature_names[fi] if feature_names else str(fi),
        "feature_idx": fi,
        "threshold": round(thresh, 4),
        "info_gain": gain,
        "samples": len(y),
        "depth": depth,
        "left":  build_tree([X[i] for i in left_mask],  [y[i] for i in left_mask],  feature_names, depth+1, max_depth),
        "right": build_tree([X[i] for i in right_mask], [y[i] for i in right_mask], feature_names, depth+1, max_depth)
    }

def predict_tree(tree, x):
    if tree["type"] == "leaf": return tree["label"]
    if x[tree["feature_idx"]] <= tree["threshold"]:
        return predict_tree(tree["left"], x)
    return predict_tree(tree["right"], x)


# ══════════════════════════════════════════════════════════════
#  ALGORITMO 6 — KNN
# ══════════════════════════════════════════════════════════════

def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        dists = [(vec_dist(x, xt), yt) for xt, yt in zip(X_train, y_train)]
        dists.sort(key=lambda t: t[0])
        neighbors = [yt for _, yt in dists[:k]]
        preds.append(Counter(neighbors).most_common(1)[0][0])
    return preds


# ══════════════════════════════════════════════════════════════
#  GERADOR DE DATASETS SINTÉTICOS
# ══════════════════════════════════════════════════════════════

def gen_blobs(n=200, k=3, noise=0.8):
    centers = [(random.uniform(-5,5), random.uniform(-5,5)) for _ in range(k)]
    data, labels = [], []
    for i in range(n):
        c = centers[i % k]
        data.append([c[0]+random.gauss(0,noise), c[1]+random.gauss(0,noise)])
        labels.append(i % k)
    return data, labels

def gen_regression(n=150, noise=15):
    x = [random.uniform(0,100) for _ in range(n)]
    y = [2.5*xi + 10 + random.gauss(0,noise) for xi in x]
    return x, y

def gen_moons(n=200, noise=0.15):
    data, labels = [], []
    for i in range(n):
        if i < n//2:
            a = math.pi * i / (n//2)
            data.append([math.cos(a)+random.gauss(0,noise), math.sin(a)+random.gauss(0,noise)])
            labels.append(0)
        else:
            a = math.pi * (i-n//2) / (n//2)
            data.append([1-math.cos(a)+random.gauss(0,noise), 1-math.sin(a)-0.5+random.gauss(0,noise)])
            labels.append(1)
    return data, labels

def gen_sales(n=100):
    """Dataset de vendas simulado"""
    records = []
    for i in range(n):
        mes    = (i % 12) + 1
        temp   = 25 + 10*math.sin(2*math.pi*mes/12) + random.gauss(0,2)
        promo  = random.choice([0,0,0,1])
        preco  = random.uniform(50,200)
        vendas = (300 + 80*math.sin(2*math.pi*mes/12) + promo*150
                  - 0.5*preco + random.gauss(0,20))
        records.append({
            "mes": mes, "temperatura": round(temp,1),
            "promocao": promo, "preco": round(preco,2),
            "vendas": round(max(0,vendas),1)
        })
    return records

def gen_iris_like(n=150):
    """Iris-like dataset"""
    classes = ["setosa","versicolor","virginica"]
    params = [
        ([5.0,3.4,1.5,0.2], [0.3,0.3,0.2,0.1]),
        ([5.9,2.8,4.3,1.3], [0.4,0.3,0.5,0.2]),
        ([6.6,3.0,5.6,2.0], [0.5,0.3,0.5,0.3]),
    ]
    data, labels = [], []
    for i in range(n):
        ci = i % 3
        mu, std = params[ci]
        row = [round(random.gauss(mu[j],std[j]),1) for j in range(4)]
        data.append(row)
        labels.append(classes[ci])
    return data, labels, ["sepal_length","sepal_width","petal_length","petal_width"]

def gen_timeseries(n=200):
    """Série temporal com tendência + sazonalidade + ruído"""
    ts = []
    for i in range(n):
        t = i/n
        trend    = 50 + 30*t
        seasonal = 15*math.sin(2*math.pi*i/52) + 8*math.sin(2*math.pi*i/12)
        noise    = random.gauss(0,3)
        anomaly  = 25 if random.random() < 0.03 else 0
        ts.append(round(trend+seasonal+noise+anomaly, 2))
    return ts


# ══════════════════════════════════════════════════════════════
#  CSV PARSER
# ══════════════════════════════════════════════════════════════

def parse_csv(content: str) -> Tuple[List[str], List[List]]:
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows: return [], []
    headers = rows[0]
    data = []
    for row in rows[1:]:
        if not any(row): continue
        parsed = []
        for cell in row:
            cell = cell.strip()
            try:
                parsed.append(float(cell))
            except:
                parsed.append(cell)
        data.append(parsed)
    return headers, data

def infer_column_types(headers, data):
    types = {}
    for i, h in enumerate(headers):
        vals = [row[i] for row in data if i < len(row)]
        numeric = sum(1 for v in vals if isinstance(v, (int,float)))
        types[h] = "numeric" if numeric > len(vals)*0.7 else "categorical"
    return types


# ══════════════════════════════════════════════════════════════
#  MÉTRICAS DE AVALIAÇÃO
# ══════════════════════════════════════════════════════════════

def classification_report(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    report = {}
    for c in classes:
        tp = sum(1 for t,p in zip(y_true,y_pred) if t==c and p==c)
        fp = sum(1 for t,p in zip(y_true,y_pred) if t!=c and p==c)
        fn = sum(1 for t,p in zip(y_true,y_pred) if t==c and p!=c)
        prec = tp/(tp+fp) if tp+fp else 0
        rec  = tp/(tp+fn) if tp+fn else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        report[str(c)] = {"precision":round(prec,3),"recall":round(rec,3),"f1":round(f1,3),"support":tp+fn}
    acc = sum(1 for t,p in zip(y_true,y_pred) if t==p)/len(y_true)
    return {"classes": report, "accuracy": round(acc,4)}

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    idx = {c:i for i,c in enumerate(classes)}
    n = len(classes)
    matrix = [[0]*n for _ in range(n)]
    for t,p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    return {"classes": [str(c) for c in classes], "matrix": matrix}


# ══════════════════════════════════════════════════════════════
#  ESTADO GLOBAL DA SESSÃO
# ══════════════════════════════════════════════════════════════

session = {
    "dataset": None,
    "headers": [],
    "col_types": {},
    "models": {},
    "history": []
}

def log_action(action, details=""):
    session["history"].append({
        "time": time.strftime("%H:%M:%S"),
        "action": action,
        "details": details
    })


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","algorithms":["kmeans","linear_regression","naive_bayes","pca","decision_tree","knn"]})

# ── Dataset Routes ─────────────────────────────────────────

@app.route("/api/dataset/generate", methods=["POST"])
def gen_dataset():
    body = request.get_json(force=True)
    name = body.get("name","blobs")
    n    = min(int(body.get("n",200)), 2000)
    k    = int(body.get("k", 3))

    if name == "blobs":
        data, labels = gen_blobs(n, k)
        headers = ["x","y","label"]
        rows = [[d[0],d[1],labels[i]] for i,d in enumerate(data)]
    elif name == "moons":
        data, labels = gen_moons(n)
        headers = ["x","y","label"]
        rows = [[d[0],d[1],labels[i]] for i,d in enumerate(data)]
    elif name == "regression":
        x, y = gen_regression(n)
        headers = ["x","y"]
        rows = [[xi,yi] for xi,yi in zip(x,y)]
    elif name == "sales":
        records = gen_sales(n)
        headers = list(records[0].keys())
        rows = [[r[h] for h in headers] for r in records]
    elif name == "iris":
        data, labels, feat_names = gen_iris_like(n)
        headers = feat_names + ["species"]
        rows = [d + [labels[i]] for i,d in enumerate(data)]
    elif name == "timeseries":
        ts = gen_timeseries(n)
        headers = ["t","value"]
        rows = [[i,v] for i,v in enumerate(ts)]
    else:
        return jsonify({"error":"Dataset desconhecido"}), 400

    session["dataset"] = rows
    session["headers"] = headers
    session["col_types"] = infer_column_types(headers, rows)
    log_action("Dataset gerado", f"{name} n={n}")

    # Quick stats preview
    numeric_cols = {h: [row[i] for row in rows if isinstance(row[i], (int,float))]
                    for i,h in enumerate(headers)}
    stats = {h: describe(v) for h,v in numeric_cols.items() if v}

    return jsonify({
        "headers": headers,
        "rows": rows[:5],  # preview
        "total": len(rows),
        "col_types": session["col_types"],
        "stats": stats
    })

@app.route("/api/dataset/upload", methods=["POST"])
def upload_dataset():
    content = request.get_data(as_text=True)
    headers, rows = parse_csv(content)
    if not rows:
        return jsonify({"error":"CSV inválido"}), 400
    session["dataset"] = rows
    session["headers"] = headers
    session["col_types"] = infer_column_types(headers, rows)
    log_action("CSV carregado", f"{len(rows)} linhas, {len(headers)} colunas")

    numeric_cols = {h: [row[i] for row in rows if i < len(row) and isinstance(row[i], (int,float))]
                    for i,h in enumerate(headers)}
    stats = {h: describe(v) for h,v in numeric_cols.items() if v}
    return jsonify({"headers":headers,"rows":rows[:5],"total":len(rows),"col_types":session["col_types"],"stats":stats})

@app.route("/api/dataset/stats")
def dataset_stats():
    if session["dataset"] is None:
        return jsonify({"error":"Nenhum dataset carregado"}), 400
    rows = session["dataset"]
    headers = session["headers"]
    numeric_cols = {h: [row[i] for row in rows if i < len(row) and isinstance(row[i],(int,float))]
                    for i,h in enumerate(headers)}
    stats = {h: describe(v) for h,v in numeric_cols.items() if v}
    corr  = correlation_matrix({h:v for h,v in numeric_cols.items() if len(v)>1})
    cat_cols = {h: dict(Counter(str(row[i]) for row in rows if i < len(row)).most_common(10))
                for i,h in enumerate(headers) if session["col_types"].get(h)=="categorical"}
    return jsonify({"stats":stats,"correlation":corr,"categorical":cat_cols,"n":len(rows)})

@app.route("/api/dataset/full")
def dataset_full():
    if session["dataset"] is None:
        return jsonify({"error":"Nenhum dataset"}), 400
    return jsonify({"headers":session["headers"],"rows":session["dataset"][:1000]})

# ── ML Routes ──────────────────────────────────────────────

@app.route("/api/ml/kmeans", methods=["POST"])
def run_kmeans():
    body = request.get_json(force=True)
    cols = body.get("features",[])
    k    = int(body.get("k",3))
    norm = body.get("normalize",True)

    rows = session["dataset"]
    if rows is None: return jsonify({"error":"Sem dataset"}), 400

    headers = session["headers"]
    idx = [headers.index(c) for c in cols if c in headers]
    if len(idx) < 2: return jsonify({"error":"Selecione >=2 features numéricas"}), 400

    data = [[row[i] for i in idx] for row in rows
            if all(isinstance(row[i],(int,float)) for i in idx)]

    if norm:
        data_sc, _, _ = zscore_scale(data)
    else:
        data_sc = data

    result = kmeans(data_sc, k)
    result["data"] = [[round(v,4) for v in row] for row in data]
    result["feature_names"] = [headers[i] for i in idx]

    # Elbow: run k=2..8
    elbow = []
    for ki in range(2, min(9, len(data))):
        r = kmeans(data_sc, ki, max_iter=30)
        elbow.append({"k": ki, "inertia": r.get("inertia",0), "silhouette": r.get("silhouette",0)})
    result["elbow"] = elbow

    session["models"]["kmeans"] = result
    log_action("K-Means executado", f"k={k} features={cols} silhouette={result['silhouette']}")
    return jsonify(result)

@app.route("/api/ml/regression", methods=["POST"])
def run_regression():
    body    = request.get_json(force=True)
    x_col   = body.get("x_col")
    y_col   = body.get("y_col")
    degree  = int(body.get("degree",1))
    rows    = session["dataset"]
    headers = session["headers"]
    if rows is None: return jsonify({"error":"Sem dataset"}), 400

    try:
        xi = headers.index(x_col)
        yi = headers.index(y_col)
    except ValueError:
        return jsonify({"error":"Coluna não encontrada"}), 400

    pairs = [(row[xi],row[yi]) for row in rows
             if isinstance(row[xi],(int,float)) and isinstance(row[yi],(int,float))]
    if len(pairs) < 3: return jsonify({"error":"Dados insuficientes"}), 400

    xs, ys = zip(*pairs)
    X = polynomial_features(list(xs), degree)
    result = linear_regression(X, list(ys))
    result["xs"] = list(xs)
    result["ys"] = list(ys)
    result["x_col"] = x_col
    result["y_col"] = y_col
    result["degree"] = degree

    # Confidence bands (±1.96 * RMSE)
    result["y_upper"] = [round(v + 1.96*result["rmse"],4) for v in result["y_pred"]]
    result["y_lower"] = [round(v - 1.96*result["rmse"],4) for v in result["y_pred"]]

    session["models"]["regression"] = result
    log_action("Regressão executada", f"grau={degree} R²={result['r2']} RMSE={result['rmse']}")
    return jsonify(result)

@app.route("/api/ml/naivebayes", methods=["POST"])
def run_naivebayes():
    body     = request.get_json(force=True)
    features = body.get("features",[])
    target   = body.get("target")
    rows     = session["dataset"]
    headers  = session["headers"]
    if rows is None: return jsonify({"error":"Sem dataset"}), 400

    try:
        feat_idx   = [headers.index(f) for f in features]
        target_idx = headers.index(target)
    except ValueError:
        return jsonify({"error":"Coluna não encontrada"}), 400

    valid = [(row, rows[i]) for i, row in enumerate(rows)
             if all(isinstance(row[fi],(int,float)) for fi in feat_idx)]
    X = [[row[fi] for fi in feat_idx] for row,_ in valid]
    y = [row[target_idx] for row,_ in valid]

    if len(set(y)) < 2: return jsonify({"error":"Target precisa ter >=2 classes"}), 400

    # Train/test split 80/20
    n = len(X)
    split = int(n*0.8)
    idx = list(range(n)); random.shuffle(idx)
    X_tr = [X[i] for i in idx[:split]]
    y_tr = [y[i] for i in idx[:split]]
    X_te = [X[i] for i in idx[split:]]
    y_te = [y[i] for i in idx[split:]]

    model = GaussianNaiveBayes().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    probas = model.predict_proba(X_te[:20])  # sample for viz

    report = classification_report(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)

    session["models"]["naivebayes"] = model
    log_action("Naive Bayes treinado", f"acc={report['accuracy']} features={features}")
    return jsonify({"report":report,"confusion_matrix":cm,"probas":probas[:10],
                    "classes":model.classes,"train_size":split,"test_size":n-split})

@app.route("/api/ml/pca", methods=["POST"])
def run_pca():
    body     = request.get_json(force=True)
    features = body.get("features",[])
    n_comp   = int(body.get("n_components",2))
    rows     = session["dataset"]
    headers  = session["headers"]
    if rows is None: return jsonify({"error":"Sem dataset"}), 400

    try:
        feat_idx = [headers.index(f) for f in features]
    except ValueError:
        return jsonify({"error":"Coluna não encontrada"}), 400

    data = [[row[fi] for fi in feat_idx] for row in rows
            if all(isinstance(row[fi],(int,float)) for fi in feat_idx)]
    if len(data) < 3: return jsonify({"error":"Dados insuficientes"}), 400

    data_sc, _, _ = zscore_scale(data)
    result = pca(data_sc, min(n_comp, len(features)))
    result["feature_names"] = features
    log_action("PCA executado", f"features={features} variância={result.get('cumulative_variance')}")
    return jsonify(result)

@app.route("/api/ml/tree", methods=["POST"])
def run_tree():
    body     = request.get_json(force=True)
    features = body.get("features",[])
    target   = body.get("target")
    max_d    = int(body.get("max_depth",4))
    rows     = session["dataset"]
    headers  = session["headers"]
    if rows is None: return jsonify({"error":"Sem dataset"}), 400

    try:
        feat_idx   = [headers.index(f) for f in features]
        target_idx = headers.index(target)
    except ValueError:
        return jsonify({"error":"Coluna não encontrada"}), 400

    valid = [row for row in rows if all(isinstance(row[fi],(int,float)) for fi in feat_idx)]
    X = [[row[fi] for fi in feat_idx] for row in valid]
    y = [row[target_idx] for row in valid]

    n = len(X); split = int(n*0.8)
    idx = list(range(n)); random.shuffle(idx)
    X_tr=[X[i] for i in idx[:split]]; y_tr=[y[i] for i in idx[:split]]
    X_te=[X[i] for i in idx[split:]]; y_te=[y[i] for i in idx[split:]]

    tree = build_tree(X_tr, y_tr, features, max_depth=max_d)
    y_pred = [predict_tree(tree, x) for x in X_te]
    report = classification_report(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)

    log_action("Árvore de Decisão", f"acc={report['accuracy']} depth={max_d}")
    return jsonify({"tree":tree,"report":report,"confusion_matrix":cm})

@app.route("/api/ml/knn", methods=["POST"])
def run_knn():
    body     = request.get_json(force=True)
    features = body.get("features",[])
    target   = body.get("target")
    k        = int(body.get("k",5))
    rows     = session["dataset"]
    headers  = session["headers"]
    if rows is None: return jsonify({"error":"Sem dataset"}), 400

    try:
        feat_idx   = [headers.index(f) for f in features]
        target_idx = headers.index(target)
    except ValueError:
        return jsonify({"error":"Coluna não encontrada"}), 400

    valid = [row for row in rows if all(isinstance(row[fi],(int,float)) for fi in feat_idx)]
    X = [[row[fi] for fi in feat_idx] for row in valid]
    y = [row[target_idx] for row in valid]

    n = len(X); split = int(n*0.8)
    idx2 = list(range(n)); random.shuffle(idx2)
    X_tr=[X[i] for i in idx2[:split]]; y_tr=[y[i] for i in idx2[:split]]
    X_te=[X[i] for i in idx2[split:]]; y_te=[y[i] for i in idx2[split:]]

    X_tr_sc,mu,std = zscore_scale(X_tr)
    X_te_sc = [[(x-mu[j])/std[j] for j,x in enumerate(row)] for row in X_te]

    y_pred = knn_predict(X_tr_sc, y_tr, X_te_sc, k)
    report = classification_report(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)

    # k-sweep
    ks_sweep = []
    for ki in [1,3,5,7,9,11]:
        if ki <= len(X_tr):
            yp = knn_predict(X_tr_sc, y_tr, X_te_sc, ki)
            ks_sweep.append({"k":ki,"acc":round(sum(1 for a,b in zip(y_te,yp) if a==b)/len(y_te),4)})

    log_action("KNN executado", f"k={k} acc={report['accuracy']}")
    return jsonify({"report":report,"confusion_matrix":cm,"k_sweep":ks_sweep})

@app.route("/api/timeseries/analyze", methods=["POST"])
def analyze_timeseries():
    body   = request.get_json(force=True)
    col    = body.get("col")
    rows   = session["dataset"]
    headers = session["headers"]
    if rows is None: return jsonify({"error":"Sem dataset"}), 400

    try:
        ci = headers.index(col)
    except ValueError:
        return jsonify({"error":"Coluna não encontrada"}), 400

    series = [row[ci] for row in rows if isinstance(row[ci],(int,float))]
    if len(series) < 10: return jsonify({"error":"Série muito curta"}), 400

    n = len(series)
    # Moving average
    w = max(3, n//20)
    ma = []
    for i in range(n):
        window = series[max(0,i-w):i+w+1]
        ma.append(round(sum(window)/len(window),4))

    # Trend (linear)
    xs = list(range(n))
    reg = linear_regression([[x] for x in xs], series)
    trend_line = reg["y_pred"]

    # Detrended + seasonal decomposition
    detrended = [series[i]-trend_line[i] for i in range(n)]

    # Autocorrelation (lag 1..20)
    acf = []
    mu = sum(series)/n
    var = sum((x-mu)**2 for x in series)/n
    for lag in range(1, min(21, n//4)):
        cov = sum((series[i]-mu)*(series[i-lag]-mu) for i in range(lag,n))/(n-lag)
        acf.append({"lag":lag,"acf":round(cov/var if var else 0,4)})

    # Anomaly detection (Z-score > 2.5)
    std = math.sqrt(var) or 1
    anomalies = [{"i":i,"value":series[i],"z":round((series[i]-mu)/std,2)}
                 for i in range(n) if abs((series[i]-mu)/std) > 2.5]

    return jsonify({
        "series": series,
        "moving_avg": ma,
        "trend_line": [round(v,4) for v in trend_line],
        "detrended": [round(v,4) for v in detrended],
        "acf": acf,
        "anomalies": anomalies,
        "trend_slope": reg["coefficients"][0] if reg.get("coefficients") else 0,
        "stats": describe(series)
    })

@app.route("/api/history")
def get_history():
    return jsonify(session["history"][-20:])

@app.route("/api/session/reset", methods=["POST"])
def reset_session():
    session["dataset"] = None
    session["headers"] = []
    session["col_types"] = {}
    session["models"] = {}
    return jsonify({"ok":True})


# ══════════════════════════════════════════════════════════════
#  HTML — DASHBOARD COMPLETO
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
#  HTML — NOVO DESIGN (Cyberpunk / Dark Terminal)
# ══════════════════════════════════════════════════════════════


HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Science Studio</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Fira+Code:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#e8eaf2;--ink2:#9ba8c0;--ink3:#4a5878;--ink4:#2a3550;
  --base:#080c14;--layer1:#0c1220;--layer2:#101828;--layer3:#141e30;
  --layer4:#1a2640;
  --cyan:#29d4f5;--green:#22e5a0;--amber:#f5a623;--rose:#f5365c;--violet:#9b6dff;--sky:#5bc8ff;
  --cyan-g:rgba(41,212,245,.1);--green-g:rgba(34,229,160,.1);--amber-g:rgba(245,166,35,.1);
  --glow-cyan:0 0 20px rgba(41,212,245,.25);
  --glow-green:0 0 20px rgba(34,229,160,.25);
  --glow-amber:0 0 20px rgba(245,166,35,.25);
  --r:3px;
  --f-ui:'Rajdhani',sans-serif;
  --f-data:'Fira Code',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:var(--base)}
body{font-family:var(--f-ui);color:var(--ink);display:flex}

/* ─── animated bg ─── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 60% 40% at 15% 50%,rgba(41,212,245,.04) 0%,transparent 60%),
    radial-gradient(ellipse 40% 60% at 85% 20%,rgba(155,109,255,.04) 0%,transparent 55%),
    radial-gradient(ellipse 50% 30% at 50% 90%,rgba(34,229,160,.03) 0%,transparent 50%);
}
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(41,212,245,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(41,212,245,.018) 1px,transparent 1px);
  background-size:60px 60px;
}

/* ─── layout ─── */
.app{position:relative;z-index:1;display:flex;width:100%;height:100vh;overflow:hidden}

/* ─── sidebar ─── */
.nav{
  width:240px;flex-shrink:0;
  background:linear-gradient(180deg,var(--layer1) 0%,var(--layer2) 100%);
  border-right:1px solid rgba(41,212,245,.08);
  display:flex;flex-direction:column;
  overflow:hidden;
}
.nav-brand{
  padding:22px 20px 18px;
  border-bottom:1px solid rgba(255,255,255,.04);
  display:flex;align-items:center;gap:14px;
}
.brand-hex{
  width:38px;height:38px;flex-shrink:0;position:relative;
  display:flex;align-items:center;justify-content:center;
}
.brand-hex svg{position:absolute;inset:0}
.brand-hex span{font-size:12px;font-weight:700;color:var(--base);position:relative;z-index:1;font-family:var(--f-data)}
.brand-name{font-size:15px;font-weight:700;letter-spacing:1.5px;color:#fff;line-height:1}
.brand-sub{font-size:10px;color:var(--ink3);font-family:var(--f-data);letter-spacing:.5px;margin-top:3px}

.nav-section{
  font-size:9px;letter-spacing:2.5px;color:var(--ink4);
  text-transform:uppercase;padding:18px 20px 8px;
  font-family:var(--f-data);
}
.nav-link{
  display:flex;align-items:center;gap:12px;
  padding:10px 20px;cursor:pointer;
  transition:all .18s;position:relative;
  font-size:13px;font-weight:500;color:var(--ink3);
  letter-spacing:.3px;
}
.nav-link::before{
  content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);
  width:2px;height:0;background:var(--cyan);border-radius:0 2px 2px 0;
  transition:height .18s;box-shadow:var(--glow-cyan);
}
.nav-link:hover{color:var(--ink);background:rgba(41,212,245,.04)}
.nav-link.on{color:#fff;background:rgba(41,212,245,.07)}
.nav-link.on::before{height:60%}
.nav-link .ico{
  width:28px;height:28px;border-radius:var(--r);
  display:flex;align-items:center;justify-content:center;
  font-size:13px;flex-shrink:0;
  background:rgba(255,255,255,.04);
  transition:background .18s;
}
.nav-link.on .ico,.nav-link:hover .ico{background:var(--cyan-g)}
.nav-link.on .ico{color:var(--cyan)}

.nav-foot{
  margin-top:auto;padding:16px 20px;
  border-top:1px solid rgba(255,255,255,.04);
}
.ds-pill{
  display:flex;align-items:center;gap:8px;
  padding:8px 12px;background:var(--layer3);
  border:1px solid rgba(41,212,245,.1);border-radius:var(--r);
  font-family:var(--f-data);
}
.ds-pulse{width:6px;height:6px;border-radius:50%;background:var(--green);flex-shrink:0;
  box-shadow:0 0 8px var(--green);animation:pulse 2.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.ds-pill-t{font-size:11px;color:var(--ink2);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ds-pill-s{font-size:9px;color:var(--ink4)}
.no-ds{color:var(--ink4);font-size:11px;font-family:var(--f-data)}

/* ─── main ─── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}

/* ─── header bar ─── */
.hbar{
  height:56px;flex-shrink:0;display:flex;align-items:center;gap:16px;
  padding:0 28px;
  background:rgba(12,18,32,.9);
  border-bottom:1px solid rgba(41,212,245,.06);
  backdrop-filter:blur(12px);
  position:relative;z-index:5;
}
.hbar-title{font-size:16px;font-weight:700;letter-spacing:.5px;color:#fff}
.hbar-sub{font-family:var(--f-data);font-size:10px;color:var(--ink4);margin-top:1px}
.hbar-tags{display:flex;gap:6px;margin-left:auto}
.tag{
  padding:3px 9px;border-radius:var(--r);
  font-family:var(--f-data);font-size:9px;letter-spacing:.8px;
  border:1px solid;text-transform:uppercase;
}
.tag-c{border-color:rgba(41,212,245,.3);color:var(--cyan);background:var(--cyan-g)}
.tag-g{border-color:rgba(34,229,160,.3);color:var(--green);background:var(--green-g)}
.tag-a{border-color:rgba(245,166,35,.3);color:var(--amber);background:var(--amber-g)}
.run-ct{font-family:var(--f-data);font-size:10px;color:var(--ink4);padding-left:10px;border-left:1px solid rgba(255,255,255,.06)}
.run-ct b{color:var(--cyan)}

/* ─── content scroll ─── */
.scroll{flex:1;overflow-y:auto;padding:24px 28px 60px;
  scrollbar-width:thin;scrollbar-color:var(--ink4) transparent}
.scroll::-webkit-scrollbar{width:3px}
.scroll::-webkit-scrollbar-thumb{background:var(--ink4);border-radius:2px}

/* ─── pages ─── */
.pg{display:none;animation:pgIn .22s ease}
.pg.on{display:block}
@keyframes pgIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* ─── grids ─── */
.row{display:grid;gap:16px;margin-bottom:16px}
.row-2{grid-template-columns:1fr 1fr}
.row-3{grid-template-columns:1fr 1fr 1fr}
.row-4{grid-template-columns:repeat(4,1fr)}
.span2{grid-column:span 2}
.span3{grid-column:1/-1}
@media(max-width:1200px){.row-4{grid-template-columns:1fr 1fr}}
@media(max-width:900px){.row-2,.row-3{grid-template-columns:1fr}}

/* ─── stat cards ─── */
.stat{
  background:var(--layer2);
  border:1px solid rgba(255,255,255,.06);
  border-radius:6px;padding:18px 20px;
  position:relative;overflow:hidden;
  cursor:default;transition:border-color .2s,transform .15s;
}
.stat:hover{transform:translateY(-2px)}
.stat-bar{position:absolute;bottom:0;left:0;right:0;height:2px}
.stat.c .stat-bar{background:linear-gradient(90deg,var(--cyan),transparent);box-shadow:0 -1px 8px var(--cyan)}
.stat.g .stat-bar{background:linear-gradient(90deg,var(--green),transparent);box-shadow:0 -1px 8px var(--green)}
.stat.a .stat-bar{background:linear-gradient(90deg,var(--amber),transparent);box-shadow:0 -1px 8px var(--amber)}
.stat.v .stat-bar{background:linear-gradient(90deg,var(--violet),transparent);box-shadow:0 -1px 8px var(--violet)}
.stat:hover.c{border-color:rgba(41,212,245,.3)}
.stat:hover.g{border-color:rgba(34,229,160,.3)}
.stat:hover.a{border-color:rgba(245,166,35,.3)}
.stat:hover.v{border-color:rgba(155,109,255,.3)}
.stat-ico{font-size:11px;color:var(--ink4);font-family:var(--f-data);letter-spacing:1px;text-transform:uppercase;margin-bottom:8px}
.stat-val{font-size:32px;font-weight:700;letter-spacing:-1px;line-height:1;font-family:var(--f-data)}
.stat.c .stat-val{color:var(--cyan)}
.stat.g .stat-val{color:var(--green)}
.stat.a .stat-val{color:var(--amber)}
.stat.v .stat-val{color:var(--violet)}

/* ─── cards ─── */
.card{
  background:var(--layer2);
  border:1px solid rgba(255,255,255,.06);
  border-radius:6px;overflow:hidden;
  transition:border-color .2s;
}
.card:hover{border-color:rgba(41,212,245,.12)}
.card-top{
  display:flex;align-items:center;gap:10px;
  padding:13px 18px;
  border-bottom:1px solid rgba(255,255,255,.04);
  background:rgba(0,0,0,.12);
}
.card-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.card-dot.c{background:var(--cyan);box-shadow:0 0 8px var(--cyan)}
.card-dot.g{background:var(--green);box-shadow:0 0 8px var(--green)}
.card-dot.a{background:var(--amber);box-shadow:0 0 8px var(--amber)}
.card-dot.v{background:var(--violet);box-shadow:0 0 8px var(--violet)}
.card-dot.r{background:var(--rose);box-shadow:0 0 8px var(--rose)}
.card-h{font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:var(--ink2)}
.card-body{padding:18px}

/* ─── data rows ─── */
.drow{
  display:flex;justify-content:space-between;align-items:center;
  padding:7px 10px;border-radius:var(--r);margin-bottom:4px;
  border-left:2px solid transparent;
  transition:all .15s;
}
.drow:hover{background:rgba(41,212,245,.04);border-left-color:rgba(41,212,245,.3)}
.dl{font-family:var(--f-data);font-size:10px;color:var(--ink3);letter-spacing:.3px}
.dv{font-family:var(--f-data);font-size:11px;font-weight:500}
.dv.c{color:var(--cyan)}.dv.g{color:var(--green)}.dv.a{color:var(--amber)}.dv.v{color:var(--violet)}

/* ─── buttons ─── */
.btn{
  display:inline-flex;align-items:center;gap:8px;
  padding:9px 20px;border-radius:var(--r);
  font-family:var(--f-ui);font-size:12px;font-weight:600;
  letter-spacing:.8px;text-transform:uppercase;
  cursor:pointer;transition:all .16s;border:1px solid transparent;
}
.btn-run{
  background:linear-gradient(135deg,rgba(41,212,245,.14),rgba(155,109,255,.1));
  border-color:rgba(41,212,245,.4);color:var(--cyan);
  box-shadow:0 0 20px rgba(41,212,245,.08);
}
.btn-run:hover{background:linear-gradient(135deg,rgba(41,212,245,.22),rgba(155,109,255,.16));
  box-shadow:0 0 28px rgba(41,212,245,.18);transform:translateY(-1px)}
.btn-run:disabled{opacity:.3;cursor:not-allowed;transform:none}
.btn-sm{background:rgba(255,255,255,.04);border-color:rgba(255,255,255,.08);
  color:var(--ink3);font-size:10px;padding:6px 12px}
.btn-sm:hover{border-color:rgba(41,212,245,.3);color:var(--cyan);background:var(--cyan-g)}
.btn-sm.on{border-color:rgba(41,212,245,.4);color:var(--cyan);background:var(--cyan-g)}
.brow{display:flex;gap:8px;flex-wrap:wrap;align-items:center}

/* ─── inputs ─── */
select,input[type=number],input[type=range]{
  background:var(--layer3);border:1px solid rgba(255,255,255,.08);border-radius:var(--r);
  color:var(--ink);font-family:var(--f-data);font-size:11px;
  padding:8px 10px;outline:none;width:100%;transition:border-color .15s;
}
select:focus,input:focus{border-color:rgba(41,212,245,.4);box-shadow:0 0 0 2px rgba(41,212,245,.06)}
select option{background:var(--layer2)}
.field{margin-bottom:13px}
.fl{display:block;font-family:var(--f-data);font-size:9px;letter-spacing:1.8px;
  color:var(--ink4);text-transform:uppercase;margin-bottom:5px}
.frow{display:flex;gap:10px;flex-wrap:wrap}
.frow .field{flex:1;min-width:80px}
.msel{background:var(--layer3);border:1px solid rgba(255,255,255,.08);border-radius:var(--r);
  padding:6px;max-height:112px;overflow-y:auto}
.msel label{display:flex;align-items:center;gap:7px;padding:4px 6px;
  font-family:var(--f-data);font-size:10px;color:var(--ink3);cursor:pointer;border-radius:var(--r);transition:all .12s}
.msel label:hover{color:var(--cyan);background:var(--cyan-g)}
.msel input[type=checkbox]{width:auto;accent-color:var(--cyan)}

/* ─── dataset chips ─── */
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}
.chip{
  padding:5px 14px;border-radius:var(--r);
  background:var(--layer3);border:1px solid rgba(255,255,255,.07);
  font-family:var(--f-data);font-size:10px;letter-spacing:.8px;
  cursor:pointer;color:var(--ink3);transition:all .15s;text-transform:uppercase;
}
.chip:hover,.chip.on{border-color:rgba(41,212,245,.4);color:var(--cyan);background:var(--cyan-g)}

/* ─── inline tabs ─── */
.itabs{display:flex;gap:2px;background:var(--layer3);
  border:1px solid rgba(255,255,255,.06);border-radius:var(--r);
  padding:3px;margin-bottom:16px}
.itab{flex:1;padding:8px 12px;border:none;border-radius:2px;
  background:transparent;color:var(--ink3);
  font-family:var(--f-data);font-size:10px;letter-spacing:1px;
  cursor:pointer;transition:all .15s;text-transform:uppercase}
.itab.on{background:var(--cyan-g);color:var(--cyan);border:1px solid rgba(41,212,245,.25)}

/* ─── tables ─── */
.twrap{overflow-x:auto;border:1px solid rgba(255,255,255,.06);border-radius:var(--r)}
table{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--f-data)}
th{background:rgba(0,0,0,.2);padding:9px 12px;text-align:left;
  font-size:9px;letter-spacing:1.5px;color:var(--ink4);text-transform:uppercase;
  border-bottom:1px solid rgba(255,255,255,.05)}
td{padding:7px 12px;border-bottom:1px solid rgba(255,255,255,.03);color:var(--ink)}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(41,212,245,.03)}
.nc{color:var(--cyan)}.cc{color:var(--violet)}

/* ─── section header ─── */
.shead{display:flex;align-items:flex-start;margin-bottom:22px;gap:14px}
.shead-line{width:3px;border-radius:2px;background:linear-gradient(180deg,var(--cyan),transparent);
  flex-shrink:0;align-self:stretch;min-height:36px}
.shead h2{font-size:18px;font-weight:700;letter-spacing:.3px;color:#fff}
.shead p{font-family:var(--f-data);font-size:10px;color:var(--ink4);margin-top:3px}
.sbadge{margin-left:auto;align-self:flex-start;padding:4px 12px;border-radius:var(--r);
  background:var(--cyan-g);border:1px solid rgba(41,212,245,.25);
  font-family:var(--f-data);font-size:9px;letter-spacing:1px;color:var(--cyan);text-transform:uppercase}

/* ─── progress ─── */
.pbar{height:4px;background:var(--layer4);border-radius:2px;overflow:hidden;margin:4px 0}
.pfill{height:100%;border-radius:2px;transition:width 1.1s cubic-bezier(.34,1.56,.64,1)}
.pfill.c{background:linear-gradient(90deg,var(--cyan),#7ef8ff)}
.pfill.g{background:linear-gradient(90deg,var(--green),#7affd4)}
.pfill.a{background:linear-gradient(90deg,var(--amber),#ffd07a)}
.pfill.v{background:linear-gradient(90deg,var(--violet),#c8aaff)}

/* ─── tree nodes ─── */
.tn{display:inline-block;padding:3px 10px;border-radius:var(--r);font-family:var(--f-data);font-size:10px;margin:2px;border:1px solid}
.tn.nd{border-color:rgba(155,109,255,.4);color:var(--violet);background:rgba(155,109,255,.07)}
.tn.lf{border-color:rgba(34,229,160,.4);color:var(--green);background:rgba(34,229,160,.07)}

/* ─── confusion matrix ─── */
.cmcell{display:flex;align-items:center;justify-content:center;
  width:52px;height:44px;border-radius:var(--r);font-family:var(--f-data);font-size:13px;font-weight:600}

/* ─── LLM ─── */
.lout{background:var(--layer3);border:1px solid rgba(255,255,255,.07);border-radius:var(--r);
  padding:16px;font-family:var(--f-data);font-size:11px;line-height:1.85;color:var(--ink);
  white-space:pre-wrap;min-height:90px;max-height:420px;overflow-y:auto}
.lout.wait{display:flex;align-items:center;gap:10px;color:var(--ink4)}
.lcur{display:inline-block;width:8px;height:14px;background:var(--cyan);
  animation:lblink .85s steps(1) infinite;vertical-align:text-bottom;margin-left:2px}
@keyframes lblink{50%{opacity:0}}
.lph{color:var(--ink4);font-size:10px;letter-spacing:.3px}
.spin{width:13px;height:13px;border:2px solid var(--ink4);border-top-color:var(--cyan);
  border-radius:50%;animation:spin .7s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}

/* ─── log ─── */
.lentry{display:flex;gap:12px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.03)}
.ltime{font-family:var(--f-data);font-size:9px;color:var(--ink4);flex-shrink:0;width:56px}
.lact{color:var(--cyan);font-family:var(--f-data);font-size:10px}
.ldet{color:var(--ink3);font-family:var(--f-data);font-size:10px;margin-top:1px}

/* ─── drop zone ─── */
.drop{border:1px dashed rgba(41,212,245,.25);border-radius:var(--r);padding:24px;text-align:center;
  cursor:pointer;transition:all .2s;font-family:var(--f-data);font-size:10px;color:var(--ink4);letter-spacing:.8px}
.drop:hover,.drop.over{border-color:rgba(41,212,245,.5);color:var(--cyan);background:var(--cyan-g)}

/* ─── SVG chart defaults ─── */
.chart-h{height:240px;position:relative}
.chart-h svg{width:100%;height:100%}

/* ─── anomaly ─── */
.anom{background:rgba(245,54,92,.08);border:1px solid rgba(245,54,92,.25);border-radius:var(--r);
  padding:2px 8px;font-family:var(--f-data);font-size:9px;color:var(--rose)}

/* ─── toast ─── */
@keyframes tIn{from{opacity:0;transform:translateX(10px)}to{opacity:1;transform:translateX(0)}}
</style>
</head>
<body>
<div class="app">

<!-- NAV -->
<nav class="nav">
  <div class="nav-brand">
    <div class="brand-hex">
      <svg viewBox="0 0 38 38" fill="none"><polygon points="19,2 36,10.5 36,27.5 19,36 2,27.5 2,10.5" fill="url(#bg)" stroke="rgba(41,212,245,.6)" stroke-width="1"/><defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="rgba(41,212,245,.3)"/><stop offset="100%" stop-color="rgba(155,109,255,.3)"/></linearGradient></defs></svg>
      <span>DS</span>
    </div>
    <div>
      <div class="brand-name">DS STUDIO</div>
      <div class="brand-sub">python · ml · llm</div>
    </div>
  </div>

  <div class="nav-section">Workspace</div>
  <div class="nav-link on" onclick="gp('home',this)"><div class="ico">⬡</div>Dashboard</div>
  <div class="nav-link" onclick="gp('data',this)"><div class="ico">◫</div>Dataset</div>
  <div class="nav-link" onclick="gp('stats',this)"><div class="ico">▤</div>Estatísticas</div>

  <div class="nav-section">Machine Learning</div>
  <div class="nav-link" onclick="gp('km',this)"><div class="ico">◎</div>K-Means++</div>
  <div class="nav-link" onclick="gp('reg',this)"><div class="ico">↗</div>Regressão OLS</div>
  <div class="nav-link" onclick="gp('clf',this)"><div class="ico">◆</div>Classificação</div>
  <div class="nav-link" onclick="gp('pca',this)"><div class="ico">⊛</div>PCA</div>

  <div class="nav-section">Análise</div>
  <div class="nav-link" onclick="gp('ts',this)"><div class="ico">∿</div>Série Temporal</div>
  <div class="nav-link" onclick="gp('llm',this)"><div class="ico">✦</div>Claude LLM</div>
  <div class="nav-link" onclick="gp('hist',this)"><div class="ico">◷</div>Histórico</div>

  <div class="nav-foot">
    <div id="nav-ds" class="no-ds">nenhum dataset</div>
  </div>
</nav>

<!-- MAIN -->
<div class="main">
  <div class="hbar">
    <div>
      <div class="hbar-title" id="ptitle">Dashboard</div>
      <div class="hbar-sub" id="psub">// selecione um dataset para começar</div>
    </div>
    <div class="hbar-tags">
      <span class="tag tag-c">Flask</span>
      <span class="tag tag-g">Python</span>
      <span class="tag tag-a">ML</span>
    </div>
    <div class="run-ct">execuções: <b id="rcount">0</b></div>
  </div>

  <div class="scroll">

<!-- ══ HOME ══ -->
<div id="pg-home" class="pg on">
  <div class="row row-4" style="margin-bottom:16px">
    <div class="stat c"><div class="stat-bar"></div><div class="stat-ico">amostras</div><div class="stat-val" id="k-rows">—</div></div>
    <div class="stat g"><div class="stat-bar"></div><div class="stat-ico">features</div><div class="stat-val" id="k-cols">—</div></div>
    <div class="stat a"><div class="stat-bar"></div><div class="stat-ico">algoritmos</div><div class="stat-val">6</div></div>
    <div class="stat v"><div class="stat-bar"></div><div class="stat-ico">execuções</div><div class="stat-val" id="k-runs">0</div></div>
  </div>

  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot c"></div><span class="card-h">Carregar Dataset</span></div>
      <div class="card-body">
        <div class="chips">
          <span class="chip on" data-ds="blobs">Blobs</span>
          <span class="chip" data-ds="moons">Moons</span>
          <span class="chip" data-ds="regression">Reg.</span>
          <span class="chip" data-ds="sales">Vendas</span>
          <span class="chip" data-ds="iris">Iris</span>
          <span class="chip" data-ds="timeseries">TimeSeries</span>
        </div>
        <div class="frow" style="margin-bottom:14px">
          <div class="field"><label class="fl">N amostras</label><input type="number" id="gn" value="300" min="50" max="2000"></div>
          <div class="field"><label class="fl">K clusters</label><input type="number" id="gk" value="3" min="2" max="8"></div>
        </div>
        <div class="brow">
          <button class="btn btn-run" onclick="genDs()">▶ Gerar</button>
          <label class="btn btn-sm" style="cursor:pointer">⬆ CSV<input type="file" accept=".csv" style="display:none" onchange="upCSV(this)"></label>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-top"><div class="card-dot g"></div><span class="card-h">Algoritmos</span></div>
      <div class="card-body">
        <div class="drow"><span class="dl">K-Means++</span><span class="dv c">Clustering</span></div>
        <div class="drow"><span class="dl">Regressão OLS</span><span class="dv g">Predição</span></div>
        <div class="drow"><span class="dl">Naive Bayes</span><span class="dv v">Classificação</span></div>
        <div class="drow"><span class="dl">PCA</span><span class="dv a">Redução Dim.</span></div>
        <div class="drow"><span class="dl">Árvore ID3</span><span class="dv c">Classificação</span></div>
        <div class="drow"><span class="dl">KNN</span><span class="dv g">Classificação</span></div>
      </div>
    </div>
  </div>

  <div class="row row-3" id="home-extra" style="display:none">
    <div class="card">
      <div class="card-top"><div class="card-dot a"></div><span class="card-h">Info do Dataset</span></div>
      <div class="card-body" id="home-info"></div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot v"></div><span class="card-h">Tipos de Colunas</span></div>
      <div class="card-body" id="home-types"></div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot c"></div><span class="card-h">Preview</span></div>
      <div class="twrap" id="home-prev"></div>
    </div>
  </div>
</div>

<!-- ══ DATA ══ -->
<div id="pg-data" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>Dataset Explorer</h2><p>// distribuições · scatter · tabela completa</p></div></div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot c"></div><span class="card-h">Histograma</span></div>
      <div class="card-body">
        <div class="field"><label class="fl">Coluna</label><select id="hcol" onchange="renderHist()"></select></div>
        <div class="chart-h" id="hist-chart"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot v"></div><span class="card-h">Scatter Plot</span></div>
      <div class="card-body">
        <div class="frow">
          <div class="field"><label class="fl">X</label><select id="scx" onchange="renderScatter()"></select></div>
          <div class="field"><label class="fl">Y</label><select id="scy" onchange="renderScatter()"></select></div>
        </div>
        <div class="chart-h" id="sc-chart"></div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-top"><div class="card-dot g"></div><span class="card-h">Tabela Completa</span></div>
    <div class="twrap" id="ftable"></div>
  </div>
</div>

<!-- ══ STATS ══ -->
<div id="pg-stats" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>Estatísticas Descritivas</h2><p>// mean · std · quartis · skewness · kurtosis · correlação</p></div>
    <button class="btn btn-sm" style="margin-left:auto" onclick="loadStats()">↺ Atualizar</button>
  </div>
  <div class="card" style="margin-bottom:16px">
    <div class="card-top"><div class="card-dot c"></div><span class="card-h">Resumo Numérico</span></div>
    <div class="twrap" id="stbl"></div>
  </div>
  <div class="card">
    <div class="card-top"><div class="card-dot a"></div><span class="card-h">Mapa de Correlação</span></div>
    <div class="card-body" id="corrmap"></div>
  </div>
</div>

<!-- ══ K-MEANS ══ -->
<div id="pg-km" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>K-Means++</h2><p>// inicialização otimizada · elbow curve · silhouette score</p></div><span class="sbadge">Unsupervised</span></div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot c"></div><span class="card-h">Parâmetros</span></div>
      <div class="card-body">
        <div class="field"><label class="fl">Features (≥2)</label><div class="msel" id="km-f"></div></div>
        <div class="frow">
          <div class="field"><label class="fl">K clusters</label><input type="number" id="km-k" value="3" min="2" max="10"></div>
          <div class="field"><label class="fl">Normalizar</label><select id="km-n"><option value="1">Z-Score</option><option value="0">Não</option></select></div>
        </div>
        <button class="btn btn-run" onclick="runKM()">▶ Executar K-Means</button>
        <div id="km-res" style="display:none;margin-top:14px">
          <div class="drow"><span class="dl">Silhouette Score</span><span class="dv g" id="km-sil">—</span></div>
          <div class="drow"><span class="dl">Inércia</span><span class="dv c" id="km-ine">—</span></div>
          <div class="drow"><span class="dl">Iterações</span><span class="dv a" id="km-itr">—</span></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot g"></div><span class="card-h">Curva Elbow</span></div>
      <div class="card-body"><div class="chart-h" id="km-elbow"></div></div>
    </div>
  </div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot v"></div><span class="card-h">Scatter dos Clusters</span></div>
      <div class="card-body"><div class="chart-h" style="height:280px" id="km-sc"></div></div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot a"></div><span class="card-h">Tamanho dos Clusters</span></div>
      <div class="card-body"><div class="chart-h" style="height:280px" id="km-dist"></div></div>
    </div>
  </div>
</div>

<!-- ══ REGRESSION ══ -->
<div id="pg-reg" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>Regressão Polinomial OLS</h2><p>// mínimos quadrados · Gauss-Jordan · bandas de confiança 95%</p></div><span class="sbadge">Supervised</span></div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot c"></div><span class="card-h">Parâmetros</span></div>
      <div class="card-body">
        <div class="frow">
          <div class="field"><label class="fl">X</label><select id="rx"></select></div>
          <div class="field"><label class="fl">Y</label><select id="ry"></select></div>
        </div>
        <div class="field"><label class="fl">Grau do polinômio</label>
          <div style="display:flex;align-items:center;gap:10px">
            <input type="range" id="rdeg" min="1" max="5" value="1" oninput="document.getElementById('rdv').textContent=this.value" style="flex:1">
            <span id="rdv" style="font-family:var(--f-data);font-size:13px;color:var(--cyan);min-width:14px">1</span>
          </div>
        </div>
        <button class="btn btn-run" onclick="runReg()">▶ Executar Regressão</button>
        <div id="reg-res" style="display:none;margin-top:14px">
          <div class="drow"><span class="dl">R²</span><span class="dv g" id="rr2">—</span></div>
          <div class="drow"><span class="dl">RMSE</span><span class="dv a" id="rrmse">—</span></div>
          <div class="drow"><span class="dl">MAE</span><span class="dv c" id="rmae">—</span></div>
          <div id="rcoef" style="font-family:var(--f-data);font-size:9px;color:var(--ink4);margin-top:8px;line-height:1.6"></div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot g"></div><span class="card-h">Regressão + IC 95%</span></div>
      <div class="card-body"><div class="chart-h" style="height:280px" id="reg-ch"></div></div>
    </div>
  </div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot a"></div><span class="card-h">Resíduos</span></div>
      <div class="card-body"><div class="chart-h" id="reg-res-ch"></div></div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot r"></div><span class="card-h">Q-Q Plot</span></div>
      <div class="card-body"><div class="chart-h" id="reg-qq-ch"></div></div>
    </div>
  </div>
</div>

<!-- ══ CLASSIFICATION ══ -->
<div id="pg-clf" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>Classificação</h2><p>// Naive Bayes Gaussiano · Árvore ID3 · K-Nearest Neighbors</p></div><span class="sbadge">Supervised</span></div>
  <div class="itabs">
    <button class="itab on" onclick="gitab('nb',this)">Naive Bayes</button>
    <button class="itab" onclick="gitab('tr',this)">Árvore ID3</button>
    <button class="itab" onclick="gitab('kn',this)">KNN</button>
  </div>

  <div id="itab-nb">
    <div class="row row-2">
      <div class="card">
        <div class="card-top"><div class="card-dot v"></div><span class="card-h">Naive Bayes Gaussiano</span></div>
        <div class="card-body">
          <div class="field"><label class="fl">Features</label><div class="msel" id="nb-f"></div></div>
          <div class="field"><label class="fl">Target</label><select id="nb-t"></select></div>
          <button class="btn btn-run" onclick="runNB()">▶ Treinar</button>
          <div id="nb-res" style="display:none;margin-top:14px">
            <div class="drow"><span class="dl">Acurácia</span><span class="dv g" id="nb-acc">—</span></div>
            <div class="drow"><span class="dl">Train / Test</span><span class="dv c" id="nb-sp">—</span></div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-top"><div class="card-dot g"></div><span class="card-h">Confusion Matrix</span></div>
        <div class="card-body" id="nb-cm"></div>
      </div>
    </div>
    <div class="card"><div class="card-top"><div class="card-dot c"></div><span class="card-h">Classification Report</span></div><div class="twrap" id="nb-rep"></div></div>
  </div>

  <div id="itab-tr" style="display:none">
    <div class="row row-2">
      <div class="card">
        <div class="card-top"><div class="card-dot a"></div><span class="card-h">Árvore de Decisão ID3</span></div>
        <div class="card-body">
          <div class="field"><label class="fl">Features</label><div class="msel" id="tr-f"></div></div>
          <div class="frow">
            <div class="field"><label class="fl">Target</label><select id="tr-t"></select></div>
            <div class="field"><label class="fl">Depth máx.</label><input type="number" id="tr-d" value="4" min="1" max="6"></div>
          </div>
          <button class="btn btn-run" onclick="runTR()">▶ Construir</button>
          <div id="tr-res" style="display:none;margin-top:14px">
            <div class="drow"><span class="dl">Acurácia</span><span class="dv g" id="tr-acc">—</span></div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-top"><div class="card-dot g"></div><span class="card-h">Confusion Matrix</span></div>
        <div class="card-body" id="tr-cm"></div>
      </div>
    </div>
    <div class="card"><div class="card-top"><div class="card-dot v"></div><span class="card-h">Estrutura da Árvore</span></div><div class="card-body" id="tr-viz" style="overflow-x:auto;font-size:11px"></div></div>
  </div>

  <div id="itab-kn" style="display:none">
    <div class="row row-2">
      <div class="card">
        <div class="card-top"><div class="card-dot c"></div><span class="card-h">K-Nearest Neighbors</span></div>
        <div class="card-body">
          <div class="field"><label class="fl">Features</label><div class="msel" id="kn-f"></div></div>
          <div class="frow">
            <div class="field"><label class="fl">Target</label><select id="kn-t"></select></div>
            <div class="field"><label class="fl">K vizinhos</label><input type="number" id="kn-k" value="5" min="1" max="21"></div>
          </div>
          <button class="btn btn-run" onclick="runKN()">▶ Executar</button>
          <div id="kn-res" style="display:none;margin-top:14px">
            <div class="drow"><span class="dl">Acurácia</span><span class="dv g" id="kn-acc">—</span></div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-top"><div class="card-dot g"></div><span class="card-h">Acurácia por K</span></div>
        <div class="card-body"><div class="chart-h" id="kn-sw"></div></div>
      </div>
    </div>
    <div class="card"><div class="card-top"><div class="card-dot v"></div><span class="card-h">Classification Report</span></div><div class="twrap" id="kn-rep"></div></div>
  </div>
</div>

<!-- ══ PCA ══ -->
<div id="pg-pca" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>PCA — Análise de Componentes Principais</h2><p>// power iteration · decomposição espectral da covariância</p></div><span class="sbadge">Unsupervised</span></div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot c"></div><span class="card-h">Configuração</span></div>
      <div class="card-body">
        <div class="field"><label class="fl">Features</label><div class="msel" id="pca-f"></div></div>
        <div class="field"><label class="fl">N componentes</label><input type="number" id="pca-n" value="2" min="2" max="6"></div>
        <button class="btn btn-run" onclick="runPCA()">▶ Executar PCA</button>
      </div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot v"></div><span class="card-h">Variância Explicada</span></div>
      <div class="card-body" id="pca-vbars"></div>
    </div>
  </div>
  <div class="card">
    <div class="card-top"><div class="card-dot g"></div><span class="card-h">Projeção 2D nos Componentes Principais</span></div>
    <div class="card-body"><div class="chart-h" style="height:300px" id="pca-proj"></div></div>
  </div>
</div>

<!-- ══ TIME SERIES ══ -->
<div id="pg-ts" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>Série Temporal</h2><p>// tendência linear · média móvel · ACF · anomalias Z-score</p></div></div>
  <div class="card" style="margin-bottom:16px">
    <div class="card-body">
      <div style="display:flex;gap:12px;align-items:flex-end">
        <div class="field" style="flex:1"><label class="fl">Coluna de valores</label><select id="ts-col"></select></div>
        <button class="btn btn-run" style="margin-bottom:13px" onclick="runTS()">▶ Analisar</button>
      </div>
    </div>
  </div>
  <div class="card" style="margin-bottom:16px">
    <div class="card-top"><div class="card-dot c"></div><span class="card-h">Série + Tendência + Média Móvel</span></div>
    <div class="card-body"><div class="chart-h" style="height:260px" id="ts-main"></div></div>
  </div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot v"></div><span class="card-h">Autocorrelação (ACF)</span></div>
      <div class="card-body"><div class="chart-h" id="ts-acf"></div></div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot r"></div><span class="card-h">Anomalias + Estatísticas</span></div>
      <div class="card-body" id="ts-anom"></div>
    </div>
  </div>
</div>

<!-- ══ LLM ══ -->
<div id="pg-llm" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>Claude LLM</h2><p>// análise generativa dos resultados do pipeline</p></div></div>
  <div class="row row-2">
    <div class="card">
      <div class="card-top"><div class="card-dot v"></div><span class="card-h">Tipo de Análise</span></div>
      <div class="card-body">
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px">
          <button class="btn btn-sm on" data-pt="interpret" onclick="spt(this)">Interpretar</button>
          <button class="btn btn-sm" data-pt="recommend" onclick="spt(this)">Recomendar</button>
          <button class="btn btn-sm" data-pt="insights" onclick="spt(this)">Insights</button>
          <button class="btn btn-sm" data-pt="report" onclick="spt(this)">Relatório</button>
          <button class="btn btn-sm" data-pt="explain" onclick="spt(this)">ELI5</button>
          <button class="btn btn-sm" data-pt="next" onclick="spt(this)">Próximos Passos</button>
        </div>
        <div class="field"><label class="fl">Contexto extra</label>
          <textarea id="lctx" style="background:var(--layer3);border:1px solid rgba(255,255,255,.08);color:var(--ink);font-family:var(--f-data);font-size:11px;padding:8px;width:100%;resize:vertical;min-height:60px;border-radius:var(--r);outline:none;transition:border-color .15s" onfocus="this.style.borderColor='rgba(41,212,245,.4)'"></textarea>
        </div>
        <button class="btn btn-run" onclick="runLLM()" id="llmbtn">✦ Executar com Claude</button>
      </div>
    </div>
    <div class="card">
      <div class="card-top"><div class="card-dot c"></div><span class="card-h">Contexto Enviado ao Claude</span></div>
      <div class="card-body"><pre id="lctxprev" style="font-family:var(--f-data);font-size:9px;color:var(--ink4);line-height:1.7;max-height:200px;overflow-y:auto;white-space:pre-wrap"></pre></div>
    </div>
  </div>
  <div class="card">
    <div class="card-top"><div class="card-dot v"></div><span class="card-h">Resposta</span></div>
    <div class="card-body"><div class="lout" id="llmout"><span class="lph">// analise um dataset e execute um algoritmo primeiro, depois peça a análise</span></div></div>
  </div>
</div>

<!-- ══ HISTORY ══ -->
<div id="pg-hist" class="pg">
  <div class="shead"><div class="shead-line"></div><div><h2>Histórico de Execuções</h2></div>
    <button class="btn btn-sm" style="margin-left:auto" onclick="loadHist()">↺</button>
  </div>
  <div class="card"><div class="card-body" id="histlog"></div></div>
</div>

  </div><!-- /scroll -->
</div><!-- /main -->
</div><!-- /app -->

<script>
var S={h:[],t:{},rows:0,name:'',mdl:{},pt:'interpret',runs:0};
var C=['#29d4f5','#22e5a0','#f5a623','#9b6dff','#f5365c','#5bc8ff','#ff6fcf','#ffe566'];
var PT={
  home:'Dashboard',data:'Dataset Explorer',stats:'Estatísticas',
  km:'K-Means++',reg:'Regressão OLS',clf:'Classificação',pca:'PCA',
  ts:'Série Temporal',llm:'Claude LLM',hist:'Histórico'
};

function gp(n,el){
  document.querySelectorAll('.pg').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.nav-link').forEach(l=>l.classList.remove('on'));
  document.getElementById('pg-'+n).classList.add('on');
  if(el)el.classList.add('on');
  document.getElementById('ptitle').textContent=PT[n]||n;
  if(n==='stats')loadStats();
  if(n==='hist')loadHist();
  if(n==='llm')updCtx();
  if(n==='data')popDataSels();
}
function gitab(n,el){
  ['nb','tr','kn'].forEach(t=>document.getElementById('itab-'+t).style.display=t===n?'block':'none');
  document.querySelectorAll('.itab').forEach(b=>b.classList.remove('on'));el.classList.add('on');
}

document.querySelectorAll('.chip').forEach(c=>c.addEventListener('click',function(){
  document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));this.classList.add('on');
}));

async function req(url,body,method){
  try{var o={method:method||(body?'POST':'GET'),headers:{'Content-Type':'application/json'}};
  if(body)o.body=JSON.stringify(body);return await(await fetch(url,o)).json();}catch(e){return{error:String(e)};}
}

async function genDs(){
  var s=document.querySelector('.chip.on');if(!s)return toast('Selecione dataset','#f5365c');
  var r=await req('/api/dataset/generate',{name:s.dataset.ds,n:+document.getElementById('gn').value||200,k:+document.getElementById('gk').value||3});
  if(r.error)return toast(r.error,'#f5365c');
  onDs(r,s.dataset.ds);
}
async function upCSV(i){
  var t=await i.files[0].text();
  var r=await fetch('/api/dataset/upload',{method:'POST',body:t}).then(x=>x.json());
  if(r.error)return toast(r.error,'#f5365c');onDs(r,i.files[0].name);
}
function onDs(r,name){
  S.h=r.headers;S.t=r.col_types;S.rows=r.total;S.name=name;
  document.getElementById('k-rows').textContent=r.total;
  document.getElementById('k-cols').textContent=r.headers.length;
  document.getElementById('nav-ds').outerHTML=`<div class="ds-pill" id="nav-ds"><div class="ds-pulse"></div><div><div class="ds-pill-t">${name}</div><div class="ds-pill-s">${r.total} × ${r.headers.length}</div></div></div>`;
  var num=Object.values(r.col_types).filter(t=>t==='numeric').length;
  document.getElementById('home-info').innerHTML=
    `<div class="drow"><span class="dl">Dataset</span><span class="dv c">${name}</span></div>`+
    `<div class="drow"><span class="dl">Amostras</span><span class="dv c">${r.total}</span></div>`+
    `<div class="drow"><span class="dl">Colunas</span><span class="dv v">${r.headers.length}</span></div>`+
    `<div class="drow"><span class="dl">Numéricas</span><span class="dv g">${num}</span></div>`;
  document.getElementById('home-types').innerHTML=r.headers.map(h=>`<div class="drow"><span class="dl">${h}</span><span class="dv ${r.col_types[h]==='numeric'?'c':'v'}">${r.col_types[h]}</span></div>`).join('');
  mkTable(r.headers,r.rows,'home-prev');
  document.getElementById('home-extra').style.display='grid';
  popAll();updCtx();toast('Dataset: '+r.total+' amostras','#22e5a0');
}
function mkTable(headers,rows,id){
  var el=document.getElementById(id);if(!el)return;
  var h='<table><thead><tr>'+headers.map(h=>`<th>${h}</th>`).join('')+'</tr></thead><tbody>';
  rows.slice(0,80).forEach(row=>{h+='<tr>'+headers.map((_,i)=>{var v=row[i];return`<td class="${typeof v==='number'?'nc':'cc'}">${typeof v==='number'?+v.toFixed(2):v}</td>`;}).join('')+'</tr>';});
  el.innerHTML=h+'</tbody></table>';
}
function popAll(){
  var num=S.h.filter(h=>S.t[h]==='numeric');var all=S.h;
  popMS('km-f',num);popMS('nb-f',num);popMS('tr-f',num);popMS('kn-f',num);popMS('pca-f',num);
  popSel('rx',num);popSel('ry',num,1);
  popSel('nb-t',all);popSel('tr-t',all);popSel('kn-t',all);
  popSel('ts-col',num);popSel('hcol',num);popSel('scx',num);popSel('scy',num,1);
}
function popDataSels(){if(S.h.length){renderHist();renderScatter();req('/api/dataset/full',null,'GET').then(r=>{if(r.rows)mkTable(r.headers,r.rows,'ftable');});}}
function popSel(id,opts,def){var el=document.getElementById(id);if(!el)return;el.innerHTML=opts.map((o,i)=>`<option value="${o}" ${i===(def||0)?'selected':''}>${o}</option>`).join('');}
function popMS(id,opts){var el=document.getElementById(id);if(!el)return;el.innerHTML=opts.map((o,i)=>`<label><input type="checkbox" value="${o}" ${i<4?'checked':''}>${o}</label>`).join('');}
function gc(id){return[...document.querySelectorAll('#'+id+' input:checked')].map(x=>x.value);}

/* ── SVG charts ── */
function W(el){return el.clientWidth||500}
function svgSc(el,pts,colors,H){
  var W2=W(el),pad=34;
  var xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
  var xn=Math.min(...xs),xx=Math.max(...xs),yn=Math.min(...ys),yx=Math.max(...ys);
  var xr=xx-xn||1,yr=yx-yn||1;
  var sx=v=>(v-xn)/xr*(W2-2*pad)+pad,sy=v=>H-((v-yn)/yr*(H-2*pad)+pad);
  var grid='';
  for(var i=1;i<4;i++){var gx=pad+i*(W2-2*pad)/4;var gy=pad+i*(H-2*pad)/4;
    grid+=`<line x1="${gx}" y1="${pad}" x2="${gx}" y2="${H-pad}" stroke="rgba(255,255,255,.04)" stroke-width="1"/>`;
    grid+=`<line x1="${pad}" y1="${gy}" x2="${W2-pad}" y2="${gy}" stroke="rgba(255,255,255,.04)" stroke-width="1"/>`;
  }
  var dots=pts.slice(0,800).map(p=>`<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3.2" fill="${colors[p.c||0]||C[0]}" opacity=".72"/>`).join('');
  el.innerHTML=`<svg viewBox="0 0 ${W2} ${H}" style="width:100%;height:${H}px">${grid}${dots}</svg>`;
}
function svgLine(el,datasets,H){
  var W2=W(el),pad=38;
  var allY=datasets.flatMap(d=>d.d);
  var yn=Math.min(...allY),yx=Math.max(...allY),yr=yx-yn||1;
  var n=datasets[0].d.length;
  var sx=i=>(i/(n-1||1))*(W2-2*pad)+pad,sy=v=>H-pad-((v-yn)/yr)*(H-2*pad);
  var ticks='';
  for(var i=0;i<=4;i++){var gy=pad+i*(H-2*pad)/4;var val=yx-(yx-yn)*i/4;
    ticks+=`<line x1="${pad}" y1="${gy}" x2="${W2-pad}" y2="${gy}" stroke="rgba(255,255,255,.04)" stroke-width="1"/>`;
    ticks+=`<text x="${pad-5}" y="${gy+4}" text-anchor="end" fill="rgba(74,88,120,.7)" font-size="9" font-family="Fira Code">${val.toFixed(1)}</text>`;
  }
  var lines=datasets.map(d=>{
    var pts2=d.d.map((v,i)=>`${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
    var apts=`${sx(0)},${H-pad} `+d.d.map((v,i)=>`${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ')+` ${sx(n-1)},${H-pad}`;
    return `<polygon points="${apts}" fill="${d.color}" opacity=".06"/><polyline points="${pts2}" fill="none" stroke="${d.color}" stroke-width="${d.w||1.6}" opacity="${d.op||.9}"/>`;
  }).join('');
  el.innerHTML=`<svg viewBox="0 0 ${W2} ${H}" style="width:100%;height:${H}px">${ticks}${lines}</svg>`;
}
function svgBar(el,labels,values,colors,H){
  var W2=W(el),pad=30;var n=labels.length;var maxV=Math.max(...values.map(Math.abs))||1;
  var bw=Math.max(6,(W2-2*pad)/n*0.55);var sp=(W2-2*pad-bw*n)/(n+1);
  var bars=labels.map((l,i)=>{
    var x=pad+sp*(i+1)+bw*i;var bh=(Math.abs(values[i])/maxV)*(H-2*pad-20);
    var y=H-pad-20-bh;var col=Array.isArray(colors)?colors[i%colors.length]:colors;
    var lbl=typeof values[i]==='number'?values[i].toFixed(2):values[i];
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw}" height="${bh.toFixed(1)}" fill="${col}" opacity=".75" rx="1"/>
<text x="${(x+bw/2).toFixed(1)}" y="${(H-pad-6).toFixed(1)}" text-anchor="middle" fill="rgba(155,168,192,.6)" font-size="8" font-family="Fira Code">${String(l).slice(0,6)}</text>`;
  }).join('');
  var ticks2='';for(var g2=1;g2<4;g2++){var gy2=pad+(H-2*pad-20)*g2/4;ticks2+=`<line x1="${pad}" y1="${gy2}" x2="${W2-pad}" y2="${gy2}" stroke="rgba(255,255,255,.04)" stroke-width="1"/>`;}
  el.innerHTML=`<svg viewBox="0 0 ${W2} ${H}" style="width:100%;height:${H}px">${ticks2}${bars}</svg>`;
}

async function renderHist(){
  var r=await req('/api/dataset/full',null,'GET');if(!r.rows)return;
  var col=document.getElementById('hcol').value,ci=r.headers.indexOf(col);
  var vals=r.rows.map(row=>row[ci]).filter(v=>typeof v==='number');
  var bins=22,mn=Math.min(...vals),mx=Math.max(...vals),step=(mx-mn)/bins||1;
  var counts=Array(bins).fill(0);vals.forEach(v=>counts[Math.min(bins-1,Math.floor((v-mn)/step))]++);
  var el=document.getElementById('hist-chart');
  svgBar(el,counts.map((_,i)=>(mn+i*step).toFixed(1)),counts,C[0],220);
}
async function renderScatter(){
  var r=await req('/api/dataset/full',null,'GET');if(!r.rows)return;
  var xi=r.headers.indexOf(document.getElementById('scx').value),yi=r.headers.indexOf(document.getElementById('scy').value);
  var pts=r.rows.slice(0,500).map(row=>({x:row[xi],y:row[yi],c:0}));
  var el=document.getElementById('sc-chart');svgSc(el,pts,[C[3]],220);
}
async function loadStats(){
  var r=await req('/api/dataset/stats',null,'GET');if(r.error)return;
  var ns=Object.keys(r.stats);var ks=['n','mean','std','min','q1','median','q3','max','skew'];
  var h='<table><thead><tr><th>Coluna</th>'+ks.map(k=>`<th>${k}</th>`).join('')+'</tr></thead><tbody>';
  ns.forEach(n=>{var s=r.stats[n];h+=`<tr><td style="color:var(--violet);font-weight:600">${n}</td>`+ks.map(k=>`<td class="nc">${s[k]??'—'}</td>`).join('')+'</tr>';});
  document.getElementById('stbl').innerHTML=h+'</tbody></table>';
  if(r.correlation&&r.correlation.names){
    var nm=r.correlation.names;var mat=r.correlation.matrix;var sz=Math.min(nm.length,8);
    var ch='<div style="overflow-x:auto"><table style="border-spacing:3px;border-collapse:separate"><tr><td></td>'+nm.slice(0,sz).map(n=>`<td style="font-family:var(--f-data);font-size:8px;color:var(--ink4);padding:2px 4px;text-align:center">${n.slice(0,6)}</td>`).join('')+'</tr>';
    for(var i=0;i<sz;i++){ch+=`<tr><td style="font-family:var(--f-data);font-size:8px;color:var(--ink4);padding:2px 6px;white-space:nowrap;text-align:right">${nm[i].slice(0,6)}</td>`;
    for(var j=0;j<sz;j++){var v=mat[nm[i]+'|'+nm[j]]||0;var a=Math.abs(v);
    var rgb=v>0?'41,212,245':'245,54,92';
    ch+=`<td style="width:46px;height:36px;background:rgba(${rgb},${a*.55+.05});border-radius:3px;text-align:center;font-family:var(--f-data);font-size:9px;color:rgba(255,255,255,${a>.25?0.9:0.3})">${v.toFixed(2)}</td>`;}ch+='</tr>';}
    document.getElementById('corrmap').innerHTML=ch+'</table></div>';
  }
}

/* ── K-Means ── */
async function runKM(){
  var features=gc('km-f'),k=+document.getElementById('km-k').value;
  if(features.length<2)return toast('Selecione ≥2 features','#f5365c');
  var r=await req('/api/ml/kmeans',{features,k,normalize:true});if(r.error)return toast(r.error,'#f5365c');
  S.runs++;document.getElementById('rcount').textContent=S.runs;document.getElementById('k-runs').textContent=S.runs;
  document.getElementById('km-res').style.display='block';
  document.getElementById('km-sil').textContent=r.silhouette;document.getElementById('km-ine').textContent=r.inertia;document.getElementById('km-itr').textContent=r.iterations;
  S.mdl.km=r;updCtx();
  var el=document.getElementById('km-elbow');
  if(r.elbow)svgLine(el,[{d:r.elbow.map(e=>e.inertia),color:C[0],w:2},{d:r.elbow.map(e=>e.silhouette*Math.max(...r.elbow.map(e2=>e2.inertia))),color:C[1],w:1.5,op:.7}],220);
  var sc=document.getElementById('km-sc');svgSc(sc,r.data.map((d,i)=>({x:d[0],y:d[1],c:r.labels[i]})),C,260);
  var di=document.getElementById('km-dist');var sk=Object.keys(r.cluster_sizes);svgBar(di,sk.map(k2=>'C'+k2),sk.map(k2=>r.cluster_sizes[k2]),C.slice(0,k),260);
  toast('K-Means sil='+r.silhouette,'#22e5a0');
}

/* ── Regression ── */
async function runReg(){
  var x_col=document.getElementById('rx').value,y_col=document.getElementById('ry').value,degree=+document.getElementById('rdeg').value||1;
  var r=await req('/api/ml/regression',{x_col,y_col,degree});if(r.error)return toast(r.error,'#f5365c');
  S.runs++;document.getElementById('rcount').textContent=S.runs;document.getElementById('k-runs').textContent=S.runs;
  document.getElementById('reg-res').style.display='block';
  document.getElementById('rr2').textContent=r.r2;document.getElementById('rrmse').textContent=r.rmse;document.getElementById('rmae').textContent=r.mae;
  document.getElementById('rcoef').textContent='β₀='+r.intercept+'  '+r.coefficients.map((c,i)=>'β'+(i+1)+'='+c).join('  ');
  S.mdl.reg=r;updCtx();
  // Main regression chart with SVG
  var el=document.getElementById('reg-ch');var W2=W(el),H=260,pad=36;
  var pts=r.xs.map((x,i)=>({x,y:r.ys[i],yp:r.y_pred[i],yu:r.y_upper[i],yl:r.y_lower[i]})).sort((a,b)=>a.x-b.x);
  var allY=[...pts.map(p=>p.y),...pts.map(p=>p.yu),...pts.map(p=>p.yl)];
  var xn=Math.min(...pts.map(p=>p.x)),xx=Math.max(...pts.map(p=>p.x));
  var yn=Math.min(...allY),yx=Math.max(...allY),xr=xx-xn||1,yr=yx-yn||1;
  var sx=v=>(v-xn)/xr*(W2-2*pad)+pad,sy=v=>H-pad-((v-yn)/yr)*(H-2*pad);
  var dots=pts.map(p=>`<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="2.5" fill="${C[0]}" opacity=".5"/>`).join('');
  var lp=pts.map(p=>`${sx(p.x).toFixed(1)},${sy(p.yp).toFixed(1)}`).join(' ');
  var up=pts.map(p=>`${sx(p.x).toFixed(1)},${sy(p.yu).toFixed(1)}`).join(' ');
  var lw=pts.map(p=>`${sx(p.x).toFixed(1)},${sy(p.yl).toFixed(1)}`).join(' ');
  var band=up+' '+pts.map(p=>`${sx(p.x).toFixed(1)},${sy(p.yl).toFixed(1)}`).reverse().join(' ');
  el.innerHTML=`<svg viewBox="0 0 ${W2} ${H}" style="width:100%;height:${H}px">
    <polygon points="${band}" fill="${C[1]}" opacity=".07"/>
    <polyline points="${up}" fill="none" stroke="${C[1]}" stroke-width=".8" stroke-dasharray="5 3" opacity=".35"/>
    <polyline points="${lw}" fill="none" stroke="${C[1]}" stroke-width=".8" stroke-dasharray="5 3" opacity=".35"/>
    ${dots}
    <polyline points="${lp}" fill="none" stroke="${C[1]}" stroke-width="2.5"/>
  </svg>`;
  // Residuals
  var re=document.getElementById('reg-res-ch');svgSc(re,r.y_pred.map((yp,i)=>({x:yp,y:r.residuals[i],c:0})),[C[2]],220);
  // Q-Q
  var qq=document.getElementById('reg-qq-ch');var s=[...r.residuals].sort((a,b)=>a-b);var nn=s.length;
  var th=s.map((_,i)=>{var p=(i+.5)/nn;return p<.5?-Math.sqrt(-2*Math.log(p)):Math.sqrt(-2*Math.log(1-p));});
  svgSc(qq,s.map((v,i)=>({x:th[i],y:v,c:0})),[C[3]],220);
  toast('Regressão R²='+r.r2,'#22e5a0');
}

/* ── Naive Bayes ── */
async function runNB(){
  var features=gc('nb-f'),target=document.getElementById('nb-t').value;
  var r=await req('/api/ml/naivebayes',{features,target});if(r.error)return toast(r.error,'#f5365c');
  S.runs++;document.getElementById('rcount').textContent=S.runs;document.getElementById('k-runs').textContent=S.runs;
  document.getElementById('nb-res').style.display='block';
  document.getElementById('nb-acc').textContent=(r.report.accuracy*100).toFixed(1)+'%';
  document.getElementById('nb-sp').textContent=r.train_size+' / '+r.test_size;
  S.mdl.nb=r;updCtx();
  renderCM(r.confusion_matrix,'nb-cm');renderRep(r.report,'nb-rep');
  toast('Naive Bayes acc='+(r.report.accuracy*100).toFixed(1)+'%','#22e5a0');
}

/* ── Tree ── */
async function runTR(){
  var features=gc('tr-f'),target=document.getElementById('tr-t').value,depth=+document.getElementById('tr-d').value||4;
  var r=await req('/api/ml/tree',{features,target,max_depth:depth});if(r.error)return toast(r.error,'#f5365c');
  S.runs++;document.getElementById('rcount').textContent=S.runs;document.getElementById('k-runs').textContent=S.runs;
  document.getElementById('tr-res').style.display='block';
  document.getElementById('tr-acc').textContent=(r.report.accuracy*100).toFixed(1)+'%';
  S.mdl.tr=r;updCtx();
  renderCM(r.confusion_matrix,'tr-cm');
  document.getElementById('tr-viz').innerHTML=treeHTML(r.tree);
  toast('Árvore acc='+(r.report.accuracy*100).toFixed(1)+'%','#22e5a0');
}
function treeHTML(n,d){d=d||0;if(!n)return'';var p='&nbsp;'.repeat(d*4);
  if(n.type==='leaf')return`${p}<span class="tn lf">▣ ${n.label} (n=${n.samples})</span><br>`;
  return`${p}<span class="tn nd">${n.feature} ≤ ${n.threshold} [ig=${n.info_gain}]</span><br>`+treeHTML(n.left,d+1)+treeHTML(n.right,d+1);
}

/* ── KNN ── */
async function runKN(){
  var features=gc('kn-f'),target=document.getElementById('kn-t').value,k=+document.getElementById('kn-k').value||5;
  var r=await req('/api/ml/knn',{features,target,k});if(r.error)return toast(r.error,'#f5365c');
  S.runs++;document.getElementById('rcount').textContent=S.runs;document.getElementById('k-runs').textContent=S.runs;
  document.getElementById('kn-res').style.display='block';
  document.getElementById('kn-acc').textContent=(r.report.accuracy*100).toFixed(1)+'%';
  S.mdl.kn=r;updCtx();
  renderRep(r.report,'kn-rep');
  var el=document.getElementById('kn-sw');svgLine(el,[{d:r.k_sweep.map(x=>x.acc),color:C[0],w:2.5}],220);
  toast('KNN acc='+(r.report.accuracy*100).toFixed(1)+'%','#22e5a0');
}

/* ── PCA ── */
async function runPCA(){
  var features=gc('pca-f'),n_components=+document.getElementById('pca-n').value||2;
  var r=await req('/api/ml/pca',{features,n_components});if(r.error)return toast(r.error,'#f5365c');
  S.runs++;document.getElementById('rcount').textContent=S.runs;document.getElementById('k-runs').textContent=S.runs;
  S.mdl.pca=r;updCtx();
  var bars='',cum=0;
  r.explained_variance_ratio.forEach((v,i)=>{cum+=v;
    bars+=`<div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;font-family:var(--f-data);font-size:10px">
        <span style="color:var(--ink3)">PC${i+1}</span>
        <span style="color:${C[i%4]}">${(v*100).toFixed(2)}%  <span style="color:var(--ink4)">cum ${(cum*100).toFixed(1)}%</span></span>
      </div>
      <div class="pbar"><div class="pfill ${['c','g','a','v'][i%4]}" style="width:${v*100}%"></div></div>
    </div>`;
  });
  document.getElementById('pca-vbars').innerHTML=bars;
  var el=document.getElementById('pca-proj');svgSc(el,r.projected.slice(0,500).map(p=>({x:p[0],y:p[1],c:0})),[C[3]],280);
  toast('PCA var='+(r.cumulative_variance.slice(-1)[0]*100).toFixed(1)+'%','#22e5a0');
}

/* ── Time Series ── */
async function runTS(){
  var col=document.getElementById('ts-col').value;
  var r=await req('/api/timeseries/analyze',{col});if(r.error)return toast(r.error,'#f5365c');
  S.runs++;document.getElementById('rcount').textContent=S.runs;document.getElementById('k-runs').textContent=S.runs;
  var el=document.getElementById('ts-main');
  svgLine(el,[{d:r.series,color:C[0],w:1.2,op:.65},{d:r.moving_avg,color:C[1],w:2.2},{d:r.trend_line,color:C[2],w:1.6,op:.8}],240);
  var acf=document.getElementById('ts-acf');svgBar(acf,r.acf.map(a=>a.lag),r.acf.map(a=>a.acf),r.acf.map(a=>Math.abs(a.acf)>.1?C[0]:C[5]),220);
  var an=document.getElementById('ts-anom');var st=r.stats;
  var ah=r.anomalies.length?r.anomalies.slice(0,7).map(a=>`<div class="drow"><span class="dl">t=${a.i}</span><span><span class="anom">z=${a.z}</span> <span class="dv" style="color:var(--rose)">${a.value}</span></span></div>`).join('')
    :'<div style="font-family:var(--f-data);font-size:10px;color:var(--ink4)">nenhuma anomalia detectada</div>';
  an.innerHTML=ah+`<div style="margin-top:10px">
    <div class="drow"><span class="dl">Média</span><span class="dv c">${st.mean}</span></div>
    <div class="drow"><span class="dl">Desvio padrão</span><span class="dv v">${st.std}</span></div>
    <div class="drow"><span class="dl">Tendência/step</span><span class="dv ${r.trend_slope>0?'g':'r'}">${(+r.trend_slope).toFixed(4)}</span></div>
  </div>`;
  toast('Série: '+r.anomalies.length+' anomalias','#22e5a0');
}

/* ── helpers ── */
function renderCM(cm,id){
  var el=document.getElementById(id);if(!el)return;
  var mx=Math.max(...cm.matrix.flat())||1;
  var h='<div style="overflow-x:auto"><div style="display:inline-block"><div style="display:flex;gap:3px;margin-bottom:3px"><div style="width:64px"></div>'+cm.classes.map(c=>`<div style="width:52px;text-align:center;font-family:var(--f-data);font-size:8px;color:var(--ink4)">${String(c).slice(0,7)}</div>`).join('')+'</div>';
  cm.matrix.forEach((row,i)=>{h+=`<div style="display:flex;gap:3px;margin-bottom:3px"><div style="width:64px;font-family:var(--f-data);font-size:8px;color:var(--ink4);display:flex;align-items:center;justify-content:flex-end;padding-right:6px">${String(cm.classes[i]).slice(0,7)}</div>`;
  row.forEach((v,j)=>{var a=v/mx;var diag=i===j;
  var bg=diag?`rgba(34,229,160,${a*.65+.1})`:`rgba(245,54,92,${a*.35+.04})`;
  var tc=diag?'#22e5a0':'#f5365c';
  h+=`<div class="cmcell" style="background:${bg};color:${tc}">${v}</div>`;});h+='</div>';});
  el.innerHTML=h+'</div></div>';
}
function renderRep(report,id){
  var el=document.getElementById(id);if(!el)return;
  var h='<table><thead><tr><th>Classe</th><th>Precisão</th><th>Recall</th><th>F1</th><th>Suporte</th></tr></thead><tbody>';
  Object.entries(report.classes).forEach(([c,m])=>{h+=`<tr><td style="color:var(--violet);font-weight:600">${c}</td><td class="nc">${m.precision}</td><td class="nc">${m.recall}</td><td class="nc">${m.f1}</td><td class="nc">${m.support}</td></tr>`;});
  h+=`<tr style="border-top:1px solid rgba(255,255,255,.06)"><td style="color:var(--green);font-weight:600">Accuracy</td><td class="nc" colspan="3" style="color:var(--green)">${report.accuracy}</td><td></td></tr>`;
  el.innerHTML=h+'</tbody></table>';
}

/* ── LLM ── */
function spt(el){document.querySelectorAll('[data-pt]').forEach(b=>b.classList.remove('on'));el.classList.add('on');S.pt=el.dataset.pt;updCtx();}
function updCtx(){
  var c=['// dataset: '+S.name+' | '+S.rows+' × '+S.h.length];
  c.push('// cols: '+S.h.slice(0,8).join(', ')+(S.h.length>8?'...':''));
  if(S.mdl.km)c.push('// kmeans: sil='+S.mdl.km.silhouette+' k='+Object.keys(S.mdl.km.cluster_sizes).length);
  if(S.mdl.reg)c.push('// regression: R²='+S.mdl.reg.r2+' RMSE='+S.mdl.reg.rmse);
  if(S.mdl.nb)c.push('// naivebayes: acc='+S.mdl.nb.report.accuracy);
  if(S.mdl.pca)c.push('// pca: var='+S.mdl.pca.explained_variance_ratio.map(v=>(v*100).toFixed(1)+'%').join(', '));
  if(S.mdl.tr)c.push('// tree: acc='+S.mdl.tr.report.accuracy);
  if(S.mdl.kn)c.push('// knn: acc='+S.mdl.kn.report.accuracy);
  document.getElementById('lctxprev').textContent=c.join('\n');
}
var LP={
  interpret:c=>`Você é cientista de dados sênior. Interprete estes resultados:\n\n${c}\n\nExplique o que significam para o negócio. Responda em português.`,
  recommend:c=>`Recomende os melhores algoritmos para explorar neste dataset:\n\n${c}\n\nJustifique tecnicamente. Responda em português.`,
  insights:c=>`Extraia 5 insights acionáveis deste dataset:\n\n${c}\n\nResponda em português.`,
  report:c=>`Gere relatório executivo (Resumo, Análise, Resultados, Recomendações):\n\n${c}\n\nResponda em português.`,
  explain:c=>`Explique para alguém sem conhecimento técnico:\n\n${c}\n\nUse analogias. Responda em português.`,
  next:c=>`5 próximos passos concretos para melhorar o modelo:\n\n${c}\n\nResponda em português.`
};
async function runLLM(){
  var ctx=document.getElementById('lctxprev').textContent;
  var extra=document.getElementById('lctx').value;if(extra)ctx+='\n\n// extra:\n'+extra;
  var out=document.getElementById('llmout'),btn=document.getElementById('llmbtn');
  btn.disabled=true;btn.textContent='⟳ Processando...';
  out.className='lout wait';out.innerHTML='<div class="spin"></div><span>Claude está analisando...</span>';
  try{
    var res=await fetch('https://api.anthropic.com/v1/messages',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'claude-sonnet-4-20250514',max_tokens:1200,messages:[{role:'user',content:LP[S.pt](ctx.slice(0,3000))}]})});
    var data=await res.json();var reply=data.content?.map(b=>b.text||'').join('')||'Sem resposta.';
    out.className='lout';out.innerHTML='';var i=0;
    var cur=document.createElement('span');cur.className='lcur';
    var tk=()=>{if(i<reply.length){out.textContent+=reply[i++];out.appendChild(cur);setTimeout(tk,6);}else cur.remove();};
    tk();toast('Claude respondeu','#22e5a0');
  }catch(e){out.className='lout';out.innerHTML='<span style="color:var(--rose)">// erro na API</span>';}
  btn.disabled=false;btn.textContent='✦ Executar com Claude';
}
async function loadHist(){
  var r=await req('/api/history',null,'GET');
  if(!Array.isArray(r)||!r.length){document.getElementById('histlog').innerHTML='<div style="color:var(--ink4);font-family:var(--f-data);font-size:10px">// sem histórico</div>';return;}
  document.getElementById('histlog').innerHTML=r.reverse().map(e=>`<div class="lentry"><div class="ltime">${e.time}</div><div><div class="lact">${e.action}</div><div class="ldet">${e.details}</div></div></div>`).join('');
}
function toast(msg,color){
  var cnt=document.querySelectorAll('.tt').length;
  var el=document.createElement('div');
  el.className='tt';
  Object.assign(el.style,{
    position:'fixed',bottom:(20+cnt*44)+'px',right:'24px',
    background:'var(--layer2)',border:'1px solid '+(color||'#22e5a0'),
    borderRadius:'3px',padding:'9px 16px',
    color:color||'#22e5a0',zIndex:9999,
    fontFamily:'Fira Code,monospace',fontSize:'11px',
    pointerEvents:'none',letterSpacing:'.3px',
    animation:'tIn .2s ease',
  });
  el.textContent=msg;document.body.appendChild(el);
  setTimeout(()=>{el.style.transition='all .3s';el.style.opacity='0';el.style.transform='translateX(8px)';setTimeout(()=>el.remove(),300);},2600);
}
fetch('/api/health').then(r=>r.json()).then(d=>{if(d.status==='ok')toast('Flask API · '+d.algorithms.length+' algoritmos','#29d4f5');});
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n" + "═"*56)
    print("  DS STUDIO v3  — Design Premium")
    print("  Rajdhani + Fira Code · SVG Charts · Dark")
    print("═"*56)
    print("  ▶  http://localhost:5000")
    print("═"*56 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
