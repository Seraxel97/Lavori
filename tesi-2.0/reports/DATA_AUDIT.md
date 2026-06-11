# Data Directory Audit Report

**Timestamp**: 2026-05-01T18:29:41+02:00
**Constraint**: disk critical 22GB liberi (22 GB required for scientific pipeline)

## System Disk Status

```
File system     Dim. Usati Dispon. Uso% Montato su
/dev/nvme2n1p2  915G  847G     22G  98% /
```

**Free available**: 22G

## Data Directory Inventory

Total size:
```
2,0G	/home/seraxel/Scrivania/Tesi_2.0/data
```

Subdirectory breakdown:

```
4,0K	/home/seraxel/Scrivania/Tesi_2.0/data/raw
529M	/home/seraxel/Scrivania/Tesi_2.0/data/derivatives
716M	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies
736M	/home/seraxel/Scrivania/Tesi_2.0/data/mock_eceo
```

## Large Files (>100MB)

```
115M /home/seraxel/Scrivania/Tesi_2.0/data/derivatives/mne-bids-pipeline/sub-05/eeg/sub-05_task-matchingpennies_proc-filt_raw.fif
357M /home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/.git/annex/objects/MJ/Q5/MD5-s373934280--cbf0332e12ffd3718327a03217d43a44/MD5-s373934280--cbf0332e12ffd3718327a03217d43a44
357M /home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-05/eeg/sub-05_task-matchingpennies_eeg.eeg
```

## Mock Data Inventory

**mock_eceo** (S-29 generated):
```
736M	/home/seraxel/Scrivania/Tesi_2.0/data/mock_eceo
```

**mock_validation_test**: not found

## Derivatives Directory (Protected)

**Status**: Contains STEP 1 output (read-only, no delete)

```
529M	/home/seraxel/Scrivania/Tesi_2.0/data/derivatives/mne-bids-pipeline/
```

## EEG MatchingPennies Dataset

```
716M	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies
```

Subdirectories:
```
4,0K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/CHANGES
4,0K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/dataset_description.json
4,0K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/participants.json
4,0K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/participants.tsv
4,0K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/task-matchingpennies_eeg.json
8,0K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/README
8,0K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/task-matchingpennies_events.json
28K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/LICENSE
36K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/code
80K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-06
80K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-07
80K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-08
80K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-09
80K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-10
80K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-11
88K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sourcedata
284K	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/stimuli
357M	/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies/sub-05
```

## Cleanup Recommendations (NO auto-delete)

**PROTECTED (do NOT delete)**:
- `data/raw/` (read-only symlink, source data)
- `data/derivatives/mne-bids-pipeline/` (STEP 1 output)

**CANDIDATES for cleanup (operator decision)**:

1. **mock_eceo/** (736M)
   - Purpose: S-29 generated mock BIDS dataset (testing only)
   - Recoverable: 736M
   - Keep if: re-running S-29 or S-52 tests
   - Remove if: tests complete, need disk

3. **eeg_matchingpennies/** (716M)
   - Purpose: main dataset (STEP 1-7 pipeline input)
   - Status: submodule (external git-annex or similar)
   - Action: check if sparse checkout can reduce size
   - WARNING: required for scientific pipeline execution

## Disk Space Estimation (post-cleanup scenarios)

**Current state**: 22G free

**Scenario A** (remove mock_eceo only):
- Estimated recovery: 0GB

**Scenario B** (remove mock_eceo + mock_validation_test):
- Estimated recovery: see Scenario A + minor test artifacts

**Scenario C** (optimize eeg_matchingpennies via sparse checkout):
- Estimated recovery: varies (contact git-annex maintainer)
- Risk: may break submodule integrity

## Recommendations

**Priority order** (for 2 GB goal):
1. Remove mock_eceo if tests complete → immediate recovery
2. Check git-annex optimization for eeg_matchingpennies
3. Verify no duplicate .fif/.vhdr/.eeg files

**Next steps**: Operator reviews this report, authorizes cleanup via separate task.

