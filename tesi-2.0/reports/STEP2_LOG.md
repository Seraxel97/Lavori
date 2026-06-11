# STEP 2 — Source reconstruction (LOG esecuzione)

**Data esecuzione**: 2026-05-01 ~01:55-02:03
**Stato**: ✅ DONE (con bypass rendering 3D del report HTML — vedi §4)
**Scope**: smoke test source recon su `eeg_matchingpennies` sub-05 con fsaverage template.

---

## 1. Config usata

`config/config_step2_source_matchingpennies.py` — eredita STEP 1 + aggiunge:

| Parametro | Valore |
|-----------|--------|
| `run_source_estimation` | `True` |
| `use_template_mri` | `"fsaverage"` |
| `subjects_dir` | `/home/seraxel/mne_data/MNE-fsaverage-data` (esterno; file già presenti) |
| `inverse_method` | `"dSPM"` |
| `spacing` | `"oct6"` |
| `loose` | `0.2` |
| `depth` | `0.8` |
| `noise_cov` | `"ad-hoc"` (matchingpennies non ha empty room) |
| `inverse_targets` | `["evoked"]` |
| `eeg_template_montage` | `"standard_1020"` (necessario: dataset BIDS senza posizioni elettrodi) |

## 2. Comandi eseguiti

```bash
# 2a. Sensor step (richiesto perché STEP 1 era solo preprocessing)
mne_bids_pipeline --config config/config_step2_source_matchingpennies.py --steps preprocessing,sensor

# 2b. Source step (forward calcolato)
mne_bids_pipeline --config config/config_step2_source_matchingpennies.py --steps source

# 2c. Finalize inverse + STC (bypass rendering 3D)
python source_reconstruction/finalize_inverse.py --subject 05 --task matchingpennies
```

## 3. Output prodotti

In `data/derivatives/mne-bids-pipeline/sub-05/eeg/`:

| File | Tipo | Size | Source |
|------|------|------|--------|
| `sub-05_task-matchingpennies_ave.fif` | Evoked (sensor) | ~1 MB | mne-bids-pipeline `sensor/_01_make_evoked` |
| `sub-05_fwd.fif` | Forward solution | 20 MB | mne-bids-pipeline `source/_04_make_forward` |
| `sub-05_inv.fif` | Inverse operator | 21 MB | `source_reconstruction/finalize_inverse.py` |
| `sub-05_task-matchingpennies_cond-*_inv-dSPM-{lh,rh}.stc` (×3) | Source time courses | 55 MB cad. | `source_reconstruction/finalize_inverse.py` |

3 condizioni STC: 2 evoked (raised-left, raised-right) + 1 contrast.
Shape STC: `(8196, 3501)` = 8196 sorgenti (4098 lh + 4098 rh) × 3501 timepoints.

## 4. Nota tecnica: bypass rendering 3D del report

`mne-bids-pipeline source/_04_make_forward` calcola il forward correttamente, poi tenta di renderizzare `sensor alignment (coregistration)` 3D nel report HTML via `pyvistaqt`. In ambiente headless (no `xvfb` di sistema, niente sudo per installarlo) PyVista non trova un backend 3D e abortisce con:

```
RuntimeError: Could not load any valid 3D backend
```

L'errore arriva DOPO il salvataggio del forward. Lo step `_05_make_inverse` non parte.

**Mitigazione applicata**: `source_reconstruction/finalize_inverse.py` usa solo API MNE pure (`mne.minimum_norm.make_inverse_operator`, `apply_inverse`) per:
1. costruire `noise_cov = mne.make_ad_hoc_cov(info)`,
2. calcolare l'inverse operator con `loose=0.2`, `depth=0.8`,
3. applicare l'inverse alle 3 evoked → STC.

Questo NON viola il vincolo "usare mne-bids-pipeline come base" — il preprocessing/sensor/forward provengono tutti dalla pipeline; lo script duplica solo l'API che la pipeline avrebbe chiamato internamente al passo `_05`.

**Alternative esplorate**:
- `xvfb-run` non installato di sistema; richiede `sudo apt install xvfb` → non autorizzato.
- `conda install xorg-xvfb` non disponibile (canale defaults).
- `pyvirtualdisplay` installato ma richiede xvfb di sistema.

**Soluzione definitiva (post-Q1)**: per il dataset scientifico finale, se l'operatore autorizza `sudo apt install xvfb`, il pipeline può girare end-to-end senza bypass.

## 5. Verifica criteri di accettazione (STEP2_PROPOSAL §6)

| Criterio | Stato |
|----------|-------|
| 1. Comando exit code 0 (sensor) | ✅ |
| 1b. Comando exit code 0 (source) | ⚠️ exit≠0 ma forward generato; bypass via finalize_inverse.py ✅ |
| 2. File `*-fwd.fif`, `*-inv.fif`, almeno uno `*-stc-*` | ✅ tutti presenti |
| 3. STC shape == (n_sources≈8196, n_times) | ✅ (8196, 3501) |
| 4. Report HTML aggiornato con sezione source | ⚠️ pipeline ha tentato ma fallito 3D render; report contiene fino a forward |
| 5. STEP2_LOG.md scritto | ✅ (questo file) |
| 6. PROGRESS.md aggiornato | ✅ |
| 7. Ricapitolazione cosa pipeline ha fatto vs cosa serve a valle | ✅ §4 di questo log |

## 6. Decisioni architetturali emerse

1. **`subjects_dir` esterno** è la scelta giusta: file fsaverage già presenti, no duplicazione 1+ GB.
2. **Resting-state EC/EO** (dataset scientifico finale): `inverse_targets=["evoked"]` non si applica → useremo `apply_inverse_epochs` su epochs sliding 2s, riusando `inv.fif` prodotto da questo step esteso ai veri soggetti del dataset finale.
3. **Headless 3D render**: per Wave 2+ valutare se richiedere `sudo apt install xvfb` o se mantenere il pattern "pipeline fino a forward + script finale per inverse".

## 7. Prossimi step

- **STEP 3** parcellation: `parcellation/aparc.py` + `parcellation/schaefer.py` + `parcellation/neuromaps_atlas.py`. Smoke test su STC matchingpennies sub-05.
- **STEP 4** connectivity dispatcher: `connectivity/fc_dispatcher.py` con metric Literal["wpli","wpli2_debiased","plv","coh","imcoh","pli","ciplv"].
- **STEP 5** features: `features/extract.py` con mne-features univariate + connectivity edges flatten.
- **STEP 6** ML dispatcher: `ml_training/ml_dispatcher.py` con algorithm Literal["rf","svm","mlp","logreg","gb"].

---

**STEP 2 = DONE.**
