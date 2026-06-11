#!/bin/bash
# Script di completamento FASE 3 — eseguire DOPO run_pipeline_n30.py --n-subjects 179
# Attende che la pipeline finisca, poi esegue ML età/sesso N=179
# Uso: bash scripts/run_fase3_post_pipeline.sh

set -euo pipefail

PYTHON="/home/seraxel/miniconda3/bin/python3.13"
HB=".planning/heartbeats_tesi/sonnet-tesi-1.json"
LOG=".planning/logs/fase3_ml_n179.log"
REPO="/home/seraxel/Scrivania/Tesi_2.0"

cd "$REPO"

echo "[$(date)] Avvio monitoraggio pipeline N=179..."

# Attende che la pipeline finisca (status=DONE nell'HB) o che X abbia 358 righe
MAX_WAIT=54000  # 15h max
INTERVAL=120    # check ogni 2 minuti
ELAPSED=0

while true; do
    HB_STATUS=$(python3 -c "
import json
try:
    d = json.load(open('$HB'))
    print(d.get('status', 'unknown'))
except:
    print('error')
" 2>/dev/null)

    X_ROWS=$(python3 -c "
import numpy as np
try:
    d = np.load('data/features/ds005385/X_aparc_plv_theta.npz')
    print(d['X'].shape[0])
except:
    print(0)
" 2>/dev/null)

    echo "[$(date)] HB status=$HB_STATUS, X rows=$X_ROWS (target=358)"

    if [[ "$HB_STATUS" == "DONE" ]] || [[ "$X_ROWS" == "358" ]]; then
        echo "[$(date)] Pipeline completata! Avvio ML età/sesso N=179..."
        break
    fi

    if [[ "$ELAPSED" -ge "$MAX_WAIT" ]]; then
        echo "[$(date)] TIMEOUT: pipeline non completata in ${MAX_WAIT}s. Uscita."
        exit 1
    fi

    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

# Esegui ML N=179
echo "[$(date)] Avvio run_ml_age_sex_n179.py --n-perm 1000 --n-boot 1000..."
"$PYTHON" scripts/run_ml_age_sex_n179.py --n-perm 1000 --n-boot 1000 2>&1 | tee "$LOG"

# Commit + push
echo "[$(date)] Commit risultati FASE 3..."
git add data/results/ds005385/ml_age_sex_n179.json reports/AGE_SEX_FASE3_N179.md 2>/dev/null || true
git commit -m "feat(tesi-fase3): step 3.7 — risultati ML età/sesso N=179 FASE 3

$(grep -E 'BA=|MAE=|R²=' "$LOG" 2>/dev/null | head -5 || echo 'see reports/AGE_SEX_FASE3_N179.md')

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" || echo "Nessun nuovo file da committare"

git push origin main || echo "Push fallito — eseguire manualmente: git push origin main"

# Ping padre
tmux send-keys -t orch-father:0 "[SONNET-TESI-FASE3-EXEC] Sprint 1 DONE (FASE 3 N=179 ML complete), commit $(git rev-parse --short HEAD), file reports/AGE_SEX_FASE3_N179.md, mcp_assisted=no" 2>/dev/null || true
sleep 1
tmux send-keys -t orch-father:0 C-m 2>/dev/null || true

echo "[$(date)] FASE 3 COMPLETATA!"
