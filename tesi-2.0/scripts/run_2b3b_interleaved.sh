#!/usr/bin/env bash
# run_2b3b_interleaved.sh — elabora soggetti uno per volta: 2b → 3b → cleanup stcs.npz
# Usato per gestire vincolo disco (~17 GB per soggetto)

set -e
SUBJECTS="${@:-sub-125 sub-157 sub-169 sub-185 sub-195}"
DERIV="data/derivatives/mne-bids-pipeline"

for SUB in $SUBJECTS; do
  echo "=========================================="
  echo "  $SUB — step 2b"
  echo "=========================================="
  python scripts/pilot_step2b_per_epoch_stc.py --subjects "$SUB"

  echo "  $SUB — step 3b"
  python scripts/pilot_step3b_per_epoch_parc.py --subjects "$SUB"

  echo "  $SUB — cleanup stcs.npz"
  rm -f "$DERIV/$SUB/eeg/${SUB}_task-RestingState_cond-EO_inv-dSPM-stcs.npz"
  rm -f "$DERIV/$SUB/eeg/${SUB}_task-RestingState_cond-EC_inv-dSPM-stcs.npz"
  echo "  $SUB cleanup done. Disk free: $(df -h /home/seraxel/Scrivania/ | tail -1 | awk '{print $4}')"
done

echo "Tutti i soggetti processati."
