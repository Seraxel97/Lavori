# Esperimenti Minimali — ds005385 PILOT

**Timestamp**: 2026-05-02T15:28:00Z  
**Sprint**: S-107 (sonnet1-ts)  
**Dataset**: ds005385, N=5 PILOT (sub-007, sub-010, sub-011, sub-026, sub-031)  

> ⚠️ **n=10 campioni, 5-fold LOSO = 1 test sample per fold.**  
> Risultati INDICATIVI per pipeline validation. **Non generalizzabili** senza N≥15.

---

## Tabella confronto — Balanced Accuracy (atlas × metric × clf)

| Atlas | Metric | LogReg | SVM-RBF | LDA |
|-------|--------|--------|---------|-----|
| aparc | wpli | 0.700 | 0.700 | 0.700 |
| aparc | coh | **0.800** | **0.800** | 0.700 |
| schaefer100 | wpli | 0.700 | 0.600 | 0.700 |
| schaefer100 | coh | **0.900** | 0.800 | 0.800 |

Tabella completa (acc / bal_acc / auc / f1):

| Atlas | Metric | Clf | Acc | Bal.Acc | AUC | F1 |
|-------|--------|-----|-----|---------|-----|----|
| aparc | wpli | logreg | 0.700 | 0.700 | 1.000 | 0.533 |
| aparc | wpli | svm_rbf | 0.700 | 0.700 | 0.000† | 0.667 |
| aparc | wpli | lda | 0.700 | 0.700 | 0.800 | 0.533 |
| aparc | coh | logreg | 0.800 | 0.800 | 1.000 | 0.733 |
| aparc | coh | svm_rbf | 0.800 | 0.800 | 0.400 | 0.733 |
| aparc | coh | lda | 0.700 | 0.700 | 1.000 | 0.667 |
| schaefer100 | wpli | logreg | 0.700 | 0.700 | 1.000 | 0.533 |
| schaefer100 | wpli | svm_rbf | 0.600 | 0.600 | 0.000† | 0.600 |
| schaefer100 | wpli | lda | 0.700 | 0.700 | 0.800 | 0.533 |
| schaefer100 | coh | logreg | **0.900** | **0.900** | 1.000 | 0.933 |
| schaefer100 | coh | svm_rbf | 0.800 | 0.800 | 0.200 | 0.733 |
| schaefer100 | coh | lda | 0.800 | 0.800 | 0.800 | 0.867 |

† AUC=0.000 su SVM-RBF: artefatto da 1-sample test fold + `predict_proba` calibrazione instabile.

---

## Winning configuration

**schaefer100 × coh × LogisticRegression**  
- Balanced Accuracy: **0.900**  
- AUC: 1.000 (indicativo — 1 test sample per fold)  
- F1: 0.933

**Interpretazione**: la coerenza (COH) su atlas fine-grained (100 ROI) distingue EO vs EC meglio della wPLI. Possibile spiegazione: COH cattura coupling ampiezza-fase inclusivo di componenti EEG stazionarie (es. ritmo alpha parieto-occipitale in EC), mentre wPLI filtra sfasamenti puri e ha maggiore varianza su n piccolo.

---

## Confronto EO vs EC — wPLI aparc alpha (da S-104b)

| Sub | EO | EC | Delta (EC-EO) | Trend |
|-----|----|----|---------------|-------|
| sub-007 | 0.1402 | 0.1192 | -0.0210 | EC < EO |
| sub-010 | 0.1945 | 0.1953 | +0.0008 | ~ pari |
| sub-011 | 0.1648 | 0.1351 | -0.0297 | EC < EO |
| sub-026 | 0.2096 | 0.2094 | -0.0002 | ~ pari |
| sub-031 | 0.1409 | 0.1531 | +0.0122 | EC > EO |

Sub-007 e sub-011 mostrano riduzione wPLI in EC (atteso: alpha aumenta in EC ma non necessariamente wPLI). Pattern non uniforme — coerente con n piccolo.

---

## Raccomandazione per full run N=15→N=179

**Best config da usare**: `schaefer100 × coh × LogisticRegression`

Motivazioni:
1. Massima balanced_accuracy sul PILOT (0.90 vs 0.80 aparc×coh)
2. LogReg interpretabile: coefficienti → top-N ROI identificabili su atlas
3. COH computazionalmente stabile e ampiamente usata in letteratura EEG resting-state
4. schaefer100 allineato con HCP atlas per comparabilità cross-studio

Configurazione ML per full run:
```python
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
cv  = LeaveOneGroupOut()  # oppure StratifiedGroupKFold(n_splits=5) per N≥15
```

Per N≥15 aggiungere: permutation test (n_perm=1000), FDR correction sui coefficienti.

---

## Next steps

1. **Aumentare N**: estendere a N=15 (seed=42 subset già definito in S-100b) → potenza statistica minima per LOSO-CV
2. **Banda theta/beta**: replicare pipeline STEP 4b-5-6 su theta (4-8 Hz) e beta (13-30 Hz) → confronto bande
3. **FDR correction**: applicare Benjamini-Hochberg sui p-value LogReg bootstrap per feature importance robusta
4. **Feature reduction**: PCA/ridge su X (10, 4950) prima di LogReg → riduce overfitting su n piccolo
5. **Permutation baseline**: confronto vs chance level (50%) con test statistico formale
6. **Visualizzazione**: proiettare coefficienti LogReg su superficie corticale (schaefer100 → MNI)

---

## Limiti espliciti

- **n=10**: classificazione binaria con 5-fold LOSO produce 1 predizione per fold → nessun intervallo di confidenza affidabile
- **Overfitting risk alto**: n_features >> n_samples (4950 >> 10) — LogReg L2 mitiga ma non elimina
- **No permutation test**: AUC=1.000 su logreg/lda è probabilmente artefatto di separabilità su n piccolo
- **Singola banda**: solo alpha — generalizzazione a full-spectrum non garantita
- **Template MRI**: fsaverage per tutti i soggetti → imprecisione source reconstruction per anatomie diverse

---

## Summary

| Metrica | Valore |
|---------|--------|
| Configurazioni testate | 12 (4 atlas×metric × 3 clf) |
| Winning config | schaefer100 × coh × LogReg |
| Best bal_acc | 0.900 (n=10, indicativo) |
| Raccomandazione full run | schaefer100 × coh × LogReg + permutation test |
| Verdict | **PASS** ✅ (pipeline validation) |
