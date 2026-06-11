# STEP 2 — Source reconstruction (PROPOSTA TECNICA)

**Data**: 2026-04-28
**Stato**: 📝 PROPOSAL — in attesa di conferma operatore prima dell'esecuzione
**Scope**: aggiungere alla pipeline ufficiale gli step `source/*` su `eeg_matchingpennies` per validare l'output source-level. Nessuna parcellazione, nessuna connettività, nessun ML.

> ⚠️ Documento di pianificazione. NON contiene codice, NON è eseguito. Serve a far approvare il piano prima di toccare il filesystem.

---

## 1. Piano STEP 2

### 1.1 Obiettivo

Verificare che `mne-bids-pipeline` produca output source-level **end-to-end** sul dataset di test `eeg_matchingpennies`, usando l'anatomia template `fsaverage` (no MRI individuale richiesta). Risultato atteso: file `*-inv.fif` (operatore inverso) + `*-evoked-stc` (source time courses sull'evoked dei contrasti) salvati in `derivatives/`.

### 1.2 Pre-requisiti tecnici (verifiche manuali pre-esecuzione)

1. **FreeSurfer template `fsaverage`** disponibile localmente. Il repo Tesi vecchio lo aveva in `/home/seraxel/mne_data/MNE-fsaverage-data/fsaverage/`. Da verificare che sia ancora lì o scaricare via `mne.datasets.fetch_fsaverage()`.
2. **`subjects_dir`** = directory padre di `fsaverage/`. Da puntare nel config STEP 2.
3. **BEM solution** per `fsaverage`: `fsaverage-5120-5120-5120-bem-sol.fif` (precomputato dal team MNE, scaricato con `fetch_fsaverage`).
4. **Source space** template: `fsaverage-ico-5-src.fif` (idem).
5. **Co-registrazione**: per `eeg_matchingpennies` non c'è MRI individuale → useremo coreg template (scaling+offset 0). `mne-bids-pipeline` lo gestisce con `use_template_mri = "fsaverage"` se attivato; altrimenti coregistrazione manuale via `mne.coreg`.

### 1.3 Sequenza esecutiva proposta

1. **Verifica fsaverage** (read-only) → log path / hash.
2. **Estendere config STEP 1** in un nuovo file `config_step2_source_matchingpennies.py` che:
   - eredita i parametri del config STEP 1 (preprocessing già validato);
   - aggiunge i parametri source (vedi §2.2);
   - punta a `subjects_dir` e usa template MRI.
3. **Run pipeline** solo `--steps source`:
   ```
   mne_bids_pipeline --config config/config_step2_source_matchingpennies.py --steps source
   ```
   Step interni eseguiti: `_01_make_bem_surfaces`, `_02_make_bem_solution`, `_03_setup_source_space`, `_04_make_forward`, `_05_make_inverse`, `_99_group_average`.
4. **Verifica output**: contare file `.fif` prodotti, controllare presenza `*-fwd.fif`, `*-inv.fif`, `*-stc` per ogni contrasto.
5. **Smoke test interpretazione**: caricare uno `*.stc` con `mne.read_source_estimate` e stampare `data.shape`, `tstep`, `vertices` per confermare formato corretto.
6. **Documentare** in `reports/STEP2_LOG.md`.

### 1.4 Cosa NON fa STEP 2 (esplicito)

- ❌ NO parcellazione (è STEP 3)
- ❌ NO connettività (è STEP 4)
- ❌ NO feature extraction (è STEP 5)
- ❌ NO ML custom (è STEP 6)
- ❌ NO uso di dataset scientifico EC/EO (resta bloccato)
- ❌ NO modifica del config STEP 1 (lo lasciamo intatto come reference)

---

## 2. Decisioni tecniche da confermare

### 2.1 Inverse method — quale fra MNE / dSPM / sLORETA / eLORETA?

| Metodo | Caratteristiche | Quando si usa | Limiti |
|--------|-----------------|----------------|--------|
| **MNE** (Minimum Norm Estimate) | Stima L2-regolarizzata; biased verso superficie | Baseline classico | Bias superficiale, scala dipende da SNR |
| **dSPM** (Dynamic SPM) | MNE normalizzato per noise covariance → unità z-score | Imaging task-based, mappe statistiche | Richiede noise cov ben stimata |
| **sLORETA** | Standardized LORETA, normalizzazione diversa | Localizzazione "zero-error" sotto noise | Bias di profondità peggiore |
| **eLORETA** | Exact LORETA, weighting più aggressivo | Connettività source-level | Più lento, in MNE ancora "experimental" su EEG |

**Raccomandazione operativa**: **`dSPM`** come default per STEP 2, perché:
- È il default ufficiale di `mne-bids-pipeline` (`inverse_method = "dSPM"` in `_config.py`);
- È quello già validato nel codice del repo Tesi vecchio (`source.py` in Tesi/) → continuità di confronto;
- Per il dataset scientifico EC/EO finale **non è necessariamente la scelta finale**: la letteratura su connettività resting-state preferisce spesso `eLORETA` (riduce field spread). Possiamo testare entrambi in STEP 4 (sensitivity analysis) ma per STEP 2 (smoke test) basta dSPM.

**Decisione richiesta**: confermo `dSPM` per STEP 2, oppure preferisci che testi tutti e 4 in parallelo già ora?

### 2.2 Parametri source da settare nel config STEP 2

| Parametro | Valore proposto | Motivazione |
|-----------|------------------|--------------|
| `inverse_method` | `"dSPM"` | vedi §2.1 |
| `spacing` | `"oct6"` | default ufficiale, ~4098 sorgenti per emisfero, bilancio risoluzione/compute |
| `noise_cov` | `"ad_hoc"` | matchingpennies non ha pre-stimulus baseline pulito sufficiente; `ad_hoc` evita estimatori vuoti (lo stesso fix usato in Tesi vecchio per RS) |
| `loose` | `0.2` | default standard EEG su superficie cortical |
| `depth` | `0.8` | depth weighting standard |
| `inverse_targets` | `["evoked"]` | DEFAULT pipeline; produce STC sull'evoked dei contrasti `raised-left vs raised-right` |
| `subjects_dir` | `/home/seraxel/mne_data/MNE-fsaverage-data` | percorso ereditato dal Tesi vecchio (DA VERIFICARE che esista) |
| `use_template_mri` | `"fsaverage"` | matchingpennies non ha MRI individuale |
| `run_source_estimation` | `True` | abilita lo step source |

**Decisione richiesta**: confermo questi 9 parametri o vuoi cambiarne alcuni?

### 2.3 Output source-level attesi

Dopo run di STEP 2 ci aspettiamo (in `data/derivatives/mne-bids-pipeline/sub-05/eeg/`):

| File | Descrizione |
|------|-------------|
| `sub-05_task-matchingpennies_fwd.fif` | Forward solution (modello propagazione) |
| `sub-05_task-matchingpennies_inv.fif` | Inverse operator |
| `sub-05_task-matchingpennies_cov.fif` | Noise covariance (se `noise_cov ≠ "ad_hoc"`) |
| `sub-05_task-matchingpennies_*-stc-*.h5` o `.stc` | Source time courses per ogni evoked condition / contrast |
| `sub-05_report.html` | Report aggiornato con sezione source |

Inoltre in `derivatives/freesurfer/` (BEM):
- `fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif`
- `fsaverage/bem/fsaverage-ico-6-src.fif` (se non già presente)

### 2.4 Limiti del dataset `eeg_matchingpennies` per source reconstruction

| Limite | Impatto STEP 2 | Impatto futuro EC/EO |
|--------|----------------|-----------------------|
| **No MRI individuale** | Si usa `fsaverage` (template anatomy) → localizzazione meno accurata di ~1-2 cm | Stesso problema su EC/EO se i soggetti non hanno MRI; standard usare fsaverage anche lì |
| **Solo 32 elettrodi BrainVision** | Spatial resolution source ridotta | EC/EO datasets (LEMON 61, ds005385 64) hanno molti più canali → source recon più affidabile |
| **Task-based (event-related)** | Pipeline produce STC su evoked → unità di analisi = "contrast", NON segmento continuo | EC/EO è resting-state: NON ha eventi, NON c'è evoked → `inverse_targets=["evoked"]` non si applica direttamente. Servirà approccio diverso: source-on-epochs sliding (vedi §3) |
| **Crop a 600s** | Stat power limitata, ma sufficiente per smoke test | Su EC/EO non si crop, si usa intera registrazione (5-10 min EC + 5-10 min EO) |
| **N=1 soggetto** | Niente group average significativo | EC/EO useremo N≥30 per statistica robusta |

### 2.5 Cosa cambia quando passeremo al dataset scientifico EC/EO

Questo è il punto critico — sul dataset finale **`mne-bids-pipeline` da solo non basta** per source-on-epochs continui:

| Aspetto | matchingpennies (STEP 2) | EC/EO scientifico (STEP futuri) |
|---------|---------------------------|----------------------------------|
| **Eventi** | `raised-left`, `raised-right` (BIDS events.tsv) | Solo annotations EC/EO (segmenti minuti); NON trial events |
| **Epoching** | `_07_make_epochs` su trigger | Va creato epoching artificiale 2s sliding (NO triggers BIDS reali) |
| **Inverse target** | `evoked` (built-in) | `epochs` source-level: pipeline NON lo produce nativamente per resting → step custom a valle |
| **Output utile** | STC per contrasto | STC per epoch (n_epochs × n_sources × n_times) — input per parcellazione e connettività |
| **Strategia** | Pipeline ufficiale `--steps source` | Pipeline ufficiale `--steps preprocessing` + step custom di source-on-epochs (modulo `source_reconstruction/`) |

**Implicazione architetturale**: la cartella `source_reconstruction/` del nostro layout NON è ridondante con `mne-bids-pipeline`. Conterrà uno script che:
- legge `*-clean_epo.fif` da `derivatives/`;
- legge `*-fwd.fif` (prodotto da pipeline) e `*-inv.fif` (prodotto da pipeline);
- applica `mne.minimum_norm.apply_inverse_epochs` → ottiene `list[SourceEstimate]` per epoch;
- salva su disco in formato compatibile con STEP 3 (parcellazione).

Questo step custom **NON viola il vincolo "usare mne-bids-pipeline come base"**: usa l'inverso e il forward generati dalla pipeline, applica solo l'API ufficiale `mne.minimum_norm` per estendere il dominio target da evoked a epochs.

---

## 3. File che intendo creare / modificare in STEP 2

### Nuovi file (da creare durante esecuzione, non ora)

| File | Tipo | Contenuto |
|------|------|-----------|
| `config/config_step2_source_matchingpennies.py` | Python config | Eredita STEP 1 + aggiunge parametri §2.2 |
| `reports/STEP2_LOG.md` | Markdown | Log esecuzione, output verificati, problemi incontrati |

### File esistenti da NON toccare in STEP 2

- `config/config_step1_matchingpennies.py` — reference baseline preprocessing
- `reports/STEP1_LOG.md` — già finalizzato
- `data/eeg_matchingpennies/` — read-only, NON ri-scaricare
- `data/derivatives/mne-bids-pipeline/sub-05/eeg/*_epo.fif` — input per source step (già presente)
- `PROGRESS.md` — da aggiornare SOLO a fine STEP 2

### File esistenti che la pipeline modificherà automaticamente

- `data/derivatives/mne-bids-pipeline/sub-05/eeg/sub-05_report.html` — pipeline ci appende sezioni source
- `data/derivatives/mne-bids-pipeline/_cache/` — cache joblib estesa con step source

### File NON creati in STEP 2 (rimando a step futuri)

- ❌ `source_reconstruction/source_on_epochs.py` (sarà creato quando passiamo a EC/EO, NON per matchingpennies)
- ❌ `parcellation/*.py` (STEP 3)
- ❌ Niente codice nei moduli `connectivity/`, `features/`, `ml_training/`, `dashboard/`

---

## 4. Rischi tecnici

| # | Rischio | Probabilità | Impatto | Mitigazione |
|---|---------|--------------|---------|--------------|
| R1 | **`fsaverage` non presente in `~/mne_data/`** sulla macchina (path ereditato dal Tesi vecchio potrebbe non esistere o essere stato spostato) | Media | Bloccante | Verificare prima dell'esecuzione: `ls ~/mne_data/MNE-fsaverage-data/fsaverage/`. Se mancante: `python -c "import mne; mne.datasets.fetch_fsaverage()"` (~250 MB download). Documentare path esatto nel config. |
| R2 | **BEM solution mancante** per fsaverage | Bassa | Pipeline genera `_01_make_bem_surfaces` + `_02_make_bem_solution` automaticamente | Confermato dal codice di pipeline `steps/source/_01_make_bem_surfaces.py` |
| R3 | **Coregistrazione template** può dare warning per landmark fiducial assenti nel BIDS di matchingpennies | Media | Warning, non bloccante | Usare `use_template_mri="fsaverage"` esplicito + `coreg = "auto"` |
| R4 | **Noise covariance vuota** (matchingpennies è task-based con baseline -200ms / 0ms): se la pipeline tenta `noise_cov="emptyroom"` fallisce per EEG | Alta se non si setta `noise_cov="ad_hoc"` | Bloccante | Forzare `noise_cov="ad_hoc"` in config (vedi §2.2) |
| R5 | **Spazio disco**: BEM + forward + inverse + STC ≈ 200-500 MB aggiuntivi | Bassa | Minore | Tesi_2.0 ha spazio sufficiente; verificare `df -h` pre-run |
| R6 | **Sostituzione path Tesi vecchio**: il `subjects_dir` punta a `/home/seraxel/mne_data/...` che è FUORI da `Tesi_2.0/` → introduce dipendenza esterna al repo | Certa | Decisione architettura | Decisione operatore: (a) lasciare `subjects_dir` esterno (più snello, condiviso con Tesi vecchio), oppure (b) copiare `fsaverage` dentro `Tesi_2.0/data/freesurfer/` (~1 GB ma self-contained) |
| R7 | **Pipeline source-on-epochs non nativa** per uso futuro RS | Certa | Architetturale | Documentato in §2.5; modulo custom verrà aggiunto SOLO quando si passa a EC/EO, non in STEP 2 |
| R8 | **Confusione con env Conda**: stiamo usando `base`, non l'env `sourcelab` del Tesi vecchio | Bassa | Tracciabilità | STEP1_LOG già documenta `base` Python 3.13.12; confermo invariato per STEP 2 |
| R9 | **Tempo esecuzione**: BEM solution + forward + inverse possono richiedere 5-15 min (single sub) | Certa | Minore | Eseguibile in foreground senza bloccare nulla |

---

## 5. Decisioni che richiedono conferma esplicita prima di eseguire STEP 2

Riassumo in checklist le decisioni — risposta `OK` / `cambio: X` per ognuna:

- [ ] **D1** — `inverse_method = "dSPM"` per STEP 2 (smoke test). Test multi-method (MNE/dSPM/sLORETA/eLORETA) rimandato a step futuro. ✅?
- [ ] **D2** — `spacing = "oct6"`, `loose = 0.2`, `depth = 0.8` (default ufficiali). ✅?
- [ ] **D3** — `noise_cov = "ad_hoc"` (necessario perché matchingpennies non ha empty room per EEG). ✅?
- [ ] **D4** — `use_template_mri = "fsaverage"` con coregistrazione automatica template (no MRI individuale). ✅?
- [ ] **D5** — `subjects_dir` esterno: `/home/seraxel/mne_data/MNE-fsaverage-data` (condiviso con altri progetti) vs interno `Tesi_2.0/data/freesurfer/` (self-contained, +1 GB). Quale preferisci?
- [ ] **D6** — Esecuzione SOLO `--steps source` su `eeg_matchingpennies sub-05` (smoke test). NIENTE source-on-epochs custom in STEP 2. ✅?
- [ ] **D7** — Se `fsaverage` non presente: autorizzo download `mne.datasets.fetch_fsaverage()` (~250 MB)? Y/N
- [ ] **D8** — Conferma di non aprire ancora la pipeline su EC/EO scientifico finché non avremo una proposta STEP 3 separata. ✅?

---

## 6. Cosa serve per dire "STEP 2 = DONE"

Criteri di accettazione (verificabili manualmente):

1. ✅ Comando `mne_bids_pipeline --config ... --steps source` termina con exit code 0.
2. ✅ Esistono i file: `*-fwd.fif`, `*-inv.fif`, almeno uno `*-stc-*` per condizione `raised-left` e `raised-right`.
3. ✅ Lo `*-stc` caricato con `mne.read_source_estimate` ha `data.shape == (n_sources, n_times)` con `n_sources` ≈ 8196 (oct6) e `n_times` corretto rispetto all'evoked window.
4. ✅ Report HTML aggiornato con sezione "Source estimates" visibile.
5. ✅ STEP2_LOG.md scritto e archiviato in `reports/`.
6. ✅ PROGRESS.md aggiornato (STEP 2 → DONE).
7. ✅ Ricapitolazione esplicita di: cosa la pipeline ha fatto vs cosa servirà costruire a valle per EC/EO (per fissare il limite e prevenire scope creep).

---

**FINE PROPOSTA. NON eseguito nulla. In attesa di tue risposte alle 8 decisioni in §5.**
