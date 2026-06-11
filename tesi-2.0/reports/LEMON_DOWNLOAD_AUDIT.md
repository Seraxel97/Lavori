# LEMON Download Audit — STEP 2.0

**Data**: 2026-06-01  
**Worker**: haiku-tesi-lemon-1  
**Status**: ⚠️ BLOCCO

---

## Tentativi

### Primario: OpenNeuro-py (ds002778, ds004504, ds003490)
- **Risultato**: FALLITO
- **Motivo**: openneuro-py non scarica dataset. Timeout o connessione. 
- **Comando tentato**: `openneuro-py download --dataset ds00XXXX --include participants.tsv --include "sub-*/eeg/*"`

### Fallback 1: EEGBCI MNE Built-in
- **Risultato**: SCARICATO, MA INVALIDO
- **Quantità**: 60 soggetti (S001–S060), run 1
- **File**: 60 × S00X/S00XR01.edf (formato EDF valido, leggibile MNE)
- **Problema**: ❌ **NO participants.tsv** con colonne `age` e `sex`
- **Causa**: EEGBCI è dataset RAW di motor imagery, metadata age/sex non disponibile in BIDS-compliant participants.tsv
- **Comando**: `mne.datasets.eegbci.load_data(subjects=range(1,61), runs=[1], path=...)`

---

## Vincolo Non Soddisfatto

Secondo SYNC step 2.0:
```
Verifica obbligatoria:
- "age" in participants.tsv columns ✓ RICHIESTO
- "sex" in participants.tsv columns ✓ RICHIESTO
- N >= 50 con age ✓ RICHIESTO
- N >= 50 con sex ✓ RICHIESTO

Se no, NON usare — segnala blocco.
```

**EEGBCI fallisce**: niente participants.tsv BIDS-compliant.

---

## Azioni Non Prese

❌ Creare participants.tsv sintetico (violererebbe "NO dati sintetici")  
❌ Dedurre age/sex da letteratura (sarebbe mock data, non BIDS-native)  
❌ Usare dati lordi senza metadata (anti-pattern HARKing: no metadata = no cross-replica validation)

---

## Prossimi Passi Proposti

1. **Contattare Orch**: request manual intervention per accesso OpenNeuro (possibile throttling IP o credential)
2. **Alternative BIDS-native**:
   - Richiedere accesso diretto S3 OpenNeuro (aws s3 ls con credenziali public)
   - FTP MPI-CBS per LEMON EEG-BIDS diretto
   - Altro dataset public BIDS-EEG (es. ds000104, ds000201)
3. **Estendere timeout**: Download su PhysioNet è lento, potrebbe servire retry con backoff

---

**Conclusione**: Step 2.0 BLOCCATO. Dataset con age+sex metadata non accessibile con metodi automatici.

---

## UNBLOCK — ds004504 via S3 (2026-06-03)
- **Dataset**: OpenNeuro ds004504 (Alzheimer EEG resting-state)
- **Fonte**: s3.amazonaws.com/openneuro.org (S3 pubblico senza credenziali)
- **N soggetti**: 88 totali, 88 con Gender+Age validi
- **Age range**: [44, 79]
- **Gender**: {F: 44, M: 44}
- **File formato**: .set (EEGLAB, leggibile mne.io.read_raw_eeglab)
- **GB su disco**: 1.5G
- **Soggetti scaricati**: 50 (sub-001..050)
- **Status**: ✅ PASS

