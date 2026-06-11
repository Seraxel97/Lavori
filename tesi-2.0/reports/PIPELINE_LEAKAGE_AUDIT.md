# Pipeline Leakage Audit — H-AUDIT-02

**Date**: 2026-05-07
**Auditor**: opus1-tesi (Opus 4.7 — complex worker)
**HEAD**: 274b53a (post H-AUDIT-01 commit)
**Scope**: verifica anti-leakage di scaling/selection/permutation nella pipeline ML N=15.

**Source files audited**
- `ml_training/ml_dispatcher.py` (run_cv, _make_pipeline, run_all_algorithms)
- `ml_training/aggregate_n15.py` (aggregate_classify_n15, _build_pipeline, _loso_cv, _permutation_p)
- `ml_training/permutation.py` (permutation_test, fdr_correction)
- `tests/test_ml_dispatcher.py` (5 algoritmi, 9 test)
- `tests/test_ml_extend.py` (extended 6-clf + multi-band, 7 test)
- `scripts/run_pipeline_n15.py` (entry point N=15)

---

## 1. Verdetto sintetico

```
LEAKAGE VERDICT: PASS
```

- **Scaling**: `StandardScaler` SEMPRE incapsulato in `sklearn.pipeline.Pipeline`, fittato solo
  sui training fold dentro il loop CV. Nessuna chiamata `scaler.fit(X)` su matrice intera trovata.
- **Feature selection**: `ABSENT` ovunque (nessun `SelectKBest`, `VarianceThreshold`, `PCA` o equivalenti).
- **Cross-validation**: `GroupKFold` per N=15 con `groups = subject_index` → split subject-aware,
  nessun overlap di soggetti train/test. `StratifiedKFold` come fallback (`groups=None`).
- **Permutation test**: shuffle delle label `y` PRIMA di ogni CV, `groups` invariato → struttura LOSO
  preservata, p-value empirico right-tailed (`P(null_BA >= observed_BA)`).
- **Hyperparameter tuning**: ASSENTE (parametri fissi `C=1.0`, `n_estimators=200`, ecc.). Nessuna
  necessità di nested CV perché non avviene selezione di hyperparametri sui dati.

Tutte le condizioni anti-leakage ML standard sono soddisfatte.

---

## 2. Audit per classificatore — posizione scaling / selector

| Classifier | Definito in | Scaling | Selector | CV usata | Verdetto |
|------------|-------------|---------|----------|----------|----------|
| logreg | `_LOCAL_CLASSIFIERS` (aggregate_n15:46) + `_CLASSIFIERS` (ml_dispatcher:35) | INSIDE_CV (Pipeline) | ABSENT | GroupKFold-15 | PASS |
| svm_rbf | `_LOCAL_CLASSIFIERS` (aggregate_n15:47) | INSIDE_CV (Pipeline) | ABSENT | GroupKFold-15 | PASS |
| svm | `_CLASSIFIERS` (ml_dispatcher:36) — alias del precedente | INSIDE_CV (Pipeline) | ABSENT | GroupKFold/StratifiedKFold | PASS |
| lda | `_LOCAL_CLASSIFIERS` (aggregate_n15:48) | INSIDE_CV (Pipeline) | ABSENT | GroupKFold-15 | PASS |
| mlp | `_CLASSIFIERS` (ml_dispatcher:37) | INSIDE_CV (Pipeline) | ABSENT | GroupKFold/StratifiedKFold | PASS |
| rf | `_CLASSIFIERS` (ml_dispatcher:38) | INSIDE_CV (Pipeline) | ABSENT | GroupKFold/StratifiedKFold | PASS |
| gb | `_CLASSIFIERS` (ml_dispatcher:39) | INSIDE_CV (Pipeline) | ABSENT | GroupKFold/StratifiedKFold | PASS |

---

## 3. Tracing del flusso CV — riga per riga

### 3.1 `aggregate_n15._loso_cv` (lines 84–102)

```python
def _loso_cv(X, y, groups, clf_name, *, n_splits) -> tuple[float, float, list[float]]:
    gkf = GroupKFold(n_splits=n_splits)
    bas: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in gkf.split(X, y, groups):
            pipe = _build_pipeline(clf_name)              # NUOVA pipeline ogni fold
            pipe.fit(X[train_idx], y[train_idx])          # FIT solo su train fold
            bas.append(float(balanced_accuracy_score(
                y[test_idx], pipe.predict(X[test_idx])    # PREDICT solo su test fold
            )))
    arr = np.array(bas)
    return float(arr.mean()), float(arr.std()), bas
```

- `gkf.split(X, y, groups)`: split subject-aware (groups = subject id, ogni soggetto compare in UN solo fold di test). ✓
- `_build_pipeline(clf_name)`: NUOVA istanza Pipeline per ogni fold (vedi 3.2).
- `pipe.fit(X[train_idx], y[train_idx])`: il Pipeline propaga internamente `scaler.fit_transform(X_train)`
  → `clf.fit(X_train_scaled, y_train)`. Lo scaler vede SOLO X_train. ✓
- `pipe.predict(X[test_idx])`: applica `scaler.transform(X_test)` (parametri imparati su train) →
  `clf.predict(X_test_scaled)`. Nessun fit su test. ✓

Verdetto: **NO leakage scaler**.

### 3.2 `aggregate_n15._build_pipeline` (lines 52–64)

```python
def _build_pipeline(clf_name: str) -> Pipeline:
    if clf_name in _LOCAL_CLASSIFIERS:
        import copy
        clf = copy.deepcopy(_LOCAL_CLASSIFIERS[clf_name])         # ← deepcopy CORRETTO
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    from ml_training.ml_dispatcher import _make_pipeline
    return _make_pipeline(clf_name)                                # ← delega (vedi 3.3)
```

Per i 3 clf locali (logreg, svm_rbf, lda): `copy.deepcopy` ad ogni call evita state sharing tra fold.
Per i 4 delegati (svm, mlp, rf, gb): eredita anti-pattern di `ml_dispatcher._make_pipeline` (vedi sotto).

### 3.3 `ml_dispatcher._make_pipeline` (lines 75–82)

```python
def _make_pipeline(algorithm: Algorithm) -> Pipeline:
    ...
    clf = _CLASSIFIERS[algorithm]                                  # ← istanza CONDIVISA (no clone)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])  # ← scaler nuovo, clf riusato
```

**Anti-pattern (NON leakage)**: `_CLASSIFIERS["logreg"]` è un singleton di modulo. Riferimenti
multipli a Pipeline puntano allo stesso oggetto `clf`. Implicazioni:

1. `Pipeline.fit` chiama `clf.fit(X_train_scaled, y_train)` che, per tutti gli sklearn classifier
   in uso (LogisticRegression, SVC, MLPClassifier, RandomForestClassifier, GradientBoostingClassifier),
   **resetta i parametri appresi** sovrascrivendo gli attributi `coef_`, `intercept_`, `tree_`, ecc.
   ⇒ NESSUN bleed-through di parametri tra fold.
2. `random_state=42` è impostato all'init quindi ogni `.fit()` riparte dallo stesso seed
   (riproducibilità preservata).
3. Race condition: in esecuzione single-thread (la pipeline N=15 è seriale) non è un problema.
   In multi-threading (es. `n_jobs=-1` di mne_connectivity, ma NON di sklearn qui) potrebbe diventare uno.

⇒ **Best-practice violation, ma NON leakage**. Raccomandazione (fuori scope di questo task):
sostituire con `clf = clone(_CLASSIFIERS[algorithm])` (`from sklearn.base import clone`).

### 3.4 `ml_dispatcher.run_cv` (lines 85–147)

```python
for train_idx, test_idx in splits:
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    pipe = _make_pipeline(algorithm)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    ba_folds.append(...)
```

Stesso schema di `_loso_cv`. Scaler INSIDE_CV. Pipeline ricreata ogni fold. ✓

---

## 4. Audit del permutation test

### 4.1 `aggregate_n15._permutation_p` (lines 105–124)

```python
def _permutation_p(X, y, groups, clf_name, observed_ba, *, n_permutations, n_splits, random_state):
    rng = np.random.default_rng(random_state)
    null_bas = np.zeros(n_permutations)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_permutations):
            ba, _, _ = _loso_cv(X, rng.permutation(y), groups, clf_name, n_splits=n_splits)
            null_bas[i] = ba
    return float(np.mean(null_bas >= observed_ba))
```

Verifiche:

- `rng.permutation(y)`: shuffle delle label (full vector, prima del CV) ✓
- `groups` **NON** permutato: la struttura LOSO è preservata, ogni soggetto resta nello stesso group.
  Questo è il pattern **corretto** per testare H0 "l'associazione (X, y) è random nel rispetto
  della struttura per-soggetto". ✓
- p-value: `mean(null_bas >= observed_ba)` — empirico right-tailed sulla null distribution. ✓
  - Un'alternativa più conservativa (Phipson & Smyth 2010) sarebbe `(sum(null >= obs) + 1) / (n + 1)`
    per evitare p-value=0 quando nessuna permutazione supera l'osservato. Nel run N=15 i p_perm=0.0000
    pubblicati (`reports/EXPERIMENTS_N15.md`) andrebbero idealmente refrasati come `p < 1/n_perm`.
    **Threat to validity** non-leakage.
- Permutation invariata tra (atlas, metric, band, classifier): `random_state` shared
  (`random_state=42` default in `aggregate_classify_n15`) ⇒ tutte le combo usano la stessa sequenza
  di permutazioni di y. Implicazioni per FDR/Bonferroni: i p-value NON sono indipendenti tra combo.
  Se si applica `fdr_correction` (`ml_training/permutation.py:99`) ai p-value aggregati, l'assunzione
  BH di indipendenza è violata. **Threat to validity** (non leakage).

### 4.2 `permutation.permutation_test` (lines 41–96)

Identico schema di `_permutation_p` ma usa `run_cv` di ml_dispatcher (StratifiedKFold se groups=None).
Comportamento atteso, nessun leakage. ✓

---

## 5. Verifiche grep (zero occorrenze pre-CV su X intero)

```bash
grep -rn "scaler.fit\|fit_transform\|SelectKBest" ml_training/ scripts/run_pipeline_n15.py
# → 0 hits
```

L'unico path scaling/fit è dentro Pipeline, dentro CV. ✓

```bash
grep -rn "scaler\|StandardScaler\|SelectKBest\|VarianceThreshold\|PCA" ml_training/*.py
# → solo definizioni dentro Pipeline (aggregate_n15:35,52,61; ml_dispatcher:14,29,75,82)
```

Nessun selector di feature presente in qualunque forma. ✓

---

## 6. Test esistenti relativi a leakage

| Test | File | Cosa verifica | Esito |
|------|------|---------------|-------|
| `test_run_cv_data_leakage_check` | `tests/test_ml_dispatcher.py:64-72` | Proxy: BA con dati corretti >= BA con X permutato (margine 15%) | Indiretto, debole |
| `test_run_cv_returns_cvresult` | `tests/test_ml_dispatcher.py:30-40` | run_cv produce CVResult con campi attesi | OK |
| `test_run_cv_ba_above_chance` | `tests/test_ml_dispatcher.py:43-48` | Con segnale, BA > 0.5 | OK |
| `test_permutation_test_null_is_near_chance` | `tests/test_ml_dispatcher.py:104-110` | Mean(null_distribution) ≈ 0.5 | OK |
| `test_six_classifiers_smoke` | `tests/test_ml_extend.py:46-66` | 6 clf su mock N=15 → 6 entries | OK |

**Gap dei test (raccomandazione, fuori scope)**: nessun test asserisce *direttamente* che lo scaler
non venga fittato su X intero. Un test ideale farebbe monkeypatch di `StandardScaler.fit` per loggare
le shape di input e verificare che `fit` veda solo `(n_train, n_features)` ad ogni call. Marker per
sprint futuro.

---

## 7. Threats to validity NON-leakage rilevati durante l'audit

Per completezza, elenco anche issue scientifici che NON sono leakage ma potrebbero confondere
l'interpretazione dei risultati N=15. Vanno discussi nel capitolo "Limiti" della tesi:

1. **Anti-pattern singleton classifier** in `ml_dispatcher._make_pipeline` (no `sklearn.base.clone`):
   non leakage ma fragile rispetto a estensioni multi-thread / multi-processo.
2. **`warnings.simplefilter("ignore")`** in `_loso_cv` (`aggregate_n15:96`) e `_permutation_p`:
   maschera `ConvergenceWarning` di LogisticRegression / MLPClassifier. Possibili modelli
   non-convergenti (specie MLP con max_iter=500 su 2278 feature × 28 train samples) sui cui risultati
   il p-value perde significato. Raccomandato logging non-silenzioso.
3. **Bootstrap CI su `ba_folds`** (`aggregate_n15:277`): bootstrap di 15 valori (uno per fold LOSO).
   Per N=15 il CI tende a essere molto largo e poco affidabile (effective sample = 15). Coerente
   con `±0.221`–`±0.298` osservati in EXPERIMENTS_N15.md.
4. **p_perm=0.0000** pubblicato: dovrebbe essere refrasato come `p < 1/n_perm` (qui `< 0.01` con
   n_perm=100, `< 0.001` con n_perm=1000). Pattern Phipson&Smyth.
5. **`random_state` condiviso tra combo per le permutazioni**: i 96 p-value derivano dalla stessa
   sequenza di shuffle, quindi NON sono indipendenti tra (atlas, metric, band, classifier). FDR-BH
   assume indipendenza: la correzione attualmente non applicata, ma se venisse applicata sarebbe
   anti-conservativa.
6. **No nested CV**: corretto ASSUMERE iperparametri fissi (no tuning ⇒ no information bleed da test
   set), ma confronto metric/atlas/band/classifier su STESSI dati = test multipli su stesso held-out.
   I 96 p_perm sono comparati ma non corretti per confronto multiplo nel report. (Coerente con §4.1
   nota su FDR.)

---

## 8. Riassunto esecutivo

```
LEAKAGE VERDICT: PASS

- Scaler: INSIDE_CV (Pipeline + fit per fold)
- Selector: ABSENT
- CV: GroupKFold-15 subject-aware
- Permutation: shuffle y prima di CV, groups invariati
- Hyperparameter tuning: ABSENT (parametri fissi)
- Nested CV: NON necessaria (no tuning)

LEAKAGE TROVATO: NESSUNO
ANTI-PATTERN MINORI (non-leakage): 6 (vedi §7)
RACCOMANDAZIONI POST-AUDIT: clone() in _make_pipeline; non-silent
warnings; rifrasare p_perm=0.0000; documentare CI bootstrap su LOSO-15.
```

---

## 9. Marker

```
[QUEUE_TASK_DONE: H-AUDIT-02]
```
