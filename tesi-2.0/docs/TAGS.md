# Tags & Taxonomy

Documentation taxonomy for Tesi_2.0 project, organized by category.

---

## Methods

Algorithms, mathematical techniques, and preprocessing approaches:

- **Source reconstruction**: inverse problem, dSPM, MNE, sLORETA, eLORETA, forward model, lead field
- **Parcellation**: atlas-based, aparc, destrieux, Schaefer-100, Schaefer-200, ROI extraction, label time-course
- **Functional connectivity metrics**: wPLI, PLV, coherence, imaginary coherence, phase locking index, wPLI2-debiased, spectral methods
- **Machine learning**: classification, regression, dimensionality reduction, feature selection, cross-validation
- **Validation**: Leave-One-Subject-Out (LOSO), GroupKFold, permutation testing, confusion matrix, AUC-ROC

---

## Tools & Libraries

Scientific computing packages and pipelines:

- **MNE ecosystem**: mne-python, mne-bids-pipeline, mne-connectivity, mne-features, mne-icalabel
- **EEG processing**: BrainVision, BIDS format, artifact rejection, ICA, bandpass filtering
- **Data science**: scikit-learn, NumPy, SciPy, pandas, matplotlib, seaborn
- **Neuroimaging**: FreeSurfer, Nilearn, Nibabel, PyVista, VTK, neuromaps
- **Infrastructure**: pre-commit, pytest, ruff, mypy, GitHub Actions, YAML configuration

---

## Topics

Domain areas and experimental paradigms:

- **EEG**: electroencephalography, resting-state, task-based, event-related potentials (ERPs)
- **Brain connectivity**: functional connectivity, spectral connectivity, source-space connectivity, cross-frequency coupling
- **Classification tasks**: eyes-closed vs eyes-open, binary classification, multi-class, feature importance
- **Resting-state analysis**: spontaneous neural activity, spectral power, connectivity patterns
- **Clinical & cognitive**: patient classification, biomarkers, neurophysiological markers

---

## Standards & Formats

Data formats, conventions, and best practices:

- **BIDS**: Brain Imaging Data Structure, standardization, data organization, metadata
- **FreeSurfer**: MRI segmentation, surface reconstruction, template brain (fsaverage), parcellation schemes
- **Python conventions**: numpy-style docstrings, type hints, linting (ruff), code quality
- **Scientific computing**: SciPy, NumPy, pandas API conventions, memory efficiency
- **Git & CI/CD**: GitHub Actions workflows, pre-commit hooks, automated testing, supply chain security

---

## Cross-References

### Vault Links

- `[[070_THESIS/MNE_PROFESSOR_LINKS]]` — canonical professor resources on MNE, source reconstruction, functional connectivity
- `[[CONNECTIVITY_METRICS]]` — detailed reference for all 7 spectral metrics (wPLI, PLV, etc.)

### Key Files

- Pipeline overview: [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)
- Benchmark design: [.planning/BENCH_DESIGN.md](../.planning/BENCH_DESIGN.md)
- Configuration base: [config/config_base.py](../config/config_base.py)
- FC dispatcher: [connectivity/fc_dispatcher.py](../connectivity/fc_dispatcher.py)
- ML dispatcher: [ml_training/ml_dispatcher.py](../ml_training/ml_dispatcher.py)

---

**Last updated**: 2026-05-01  
**Scope**: Tesi_2.0 — EEG Source-Space FC Pipeline
