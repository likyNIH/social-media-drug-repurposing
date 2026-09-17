#!/bin/bash
#SBATCH --job-name=drug_extraction
#SBATCH --partition=extended_gpu
#SBATCH --account=ncats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --output=/ncats/users/liky/social_media_data/logs/drug_extraction_%j.out
#SBATCH --error=/ncats/users/liky/social_media_data/logs/drug_extraction_%j.err
#SBATCH --signal=B:SIGUSR1@120

# Same GPU/model resources as scleroderma_drug_extraction.sh (gemma3-27b
# needs 4 GPUs regardless of workload size), but on extended_gpu (5-day
# limit, confirmed to exist alongside extended_cpu) instead of normal_gpu,
# since this job is much bigger than the original scleroderma-only run:
# 133,028 filtered documents across 20 communities vs. 4,598 for one. Fewer,
# longer segments matter more here than for the CPU jobs, since reloading
# the 27B model (~94s init) is a fixed cost paid on every chain handoff.
#
# Resumable at both the community level (run_drug_extraction.py skips any
# community whose *_drug_signals.jsonl already exists) and the post level
# within a community (writes to a .tmp file, flushed per post, resumes from
# a partial file on restart) -- same pattern the scraper needed for the
# same reason: a multi-day job over a large corpus needs to survive
# crashes/time-limits without redoing already-completed work.
#
# Chain-resubmission covers both the graceful time-limit signal AND an
# actual crash, same as submit_scrape_rare_disease_subs.sh: a crash within
# the first 60s is treated as a likely systemic bug (won't auto-loop), a
# crash after that is treated as a recoverable one-off (auto-resubmits).
#
# Usage: sbatch submit_drug_extraction.sh

set -o pipefail
# NOT set -u: the test-llama conda env's own activation hook
# (activate-binutils_linux-64.sh) references an unbound ADDR2LINE variable,
# which -u treats as fatal, killing the job before it ever reaches
# `conda activate` -- confirmed both by a real failed run (job 10025988,
# died in ~5s with exactly this error) and by directly reproducing it in a
# plain shell with no SLURM/GPU involved at all. scleroderma_drug_extraction.sh
# never used set -u either, almost certainly for this same reason.

SCRIPT_PATH="/ncats/users/liky/social_media_data/submit_drug_extraction.sh"
MIN_RUNTIME_FOR_AUTO_RESTART_S=60

on_time_limit() {
    echo "Time limit approaching — stopping current run and queuing continuation job..."
    kill -TERM "$PY_PID" 2>/dev/null
    wait "$PY_PID" 2>/dev/null
    NEXT_JOBID=$(sbatch --parsable "$SCRIPT_PATH")
    echo "Queued continuation job: $NEXT_JOBID"
    exit 0
}
trap on_time_limit SIGUSR1

mkdir -p /ncats/users/liky/social_media_data/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate test-llama

START_TIME=$(date +%s)
python /ncats/users/liky/social_media_data/run_drug_extraction.py &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?
ELAPSED=$(( $(date +%s) - START_TIME ))

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Finished cleanly (all communities extracted)."
elif [ "$ELAPSED" -lt "$MIN_RUNTIME_FOR_AUTO_RESTART_S" ]; then
    echo "Python exited with code $EXIT_CODE after only ${ELAPSED}s — too fast to be a rare data edge" \
         "case, likely a systemic bug. NOT auto-resubmitting; investigate before rerunning manually."
    exit "$EXIT_CODE"
else
    echo "Python exited with code $EXIT_CODE after ${ELAPSED}s — treating as a recoverable crash," \
         "queuing continuation job..."
    NEXT_JOBID=$(sbatch --parsable "$SCRIPT_PATH")
    echo "Queued continuation job: $NEXT_JOBID"
fi
