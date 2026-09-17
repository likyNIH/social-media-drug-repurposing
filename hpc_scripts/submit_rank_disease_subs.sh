#!/bin/bash
#SBATCH --job-name=rank_disease_subs
#SBATCH --output=/ncats/users/liky/social_media_data/logs/rank_disease_subs_%j.out
#SBATCH --error=/ncats/users/liky/social_media_data/logs/rank_disease_subs_%j.err
#SBATCH --partition=quick_cpu
#SBATCH --account=ncats
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --signal=B:SIGUSR1@120

# Submits rank_disease_subreddits.py as a self-resubmitting chain job.
# Runtime is ~2 hours for the full ~6000-disease list (network-bound, single
# core is enough); --time=04:00:00 leaves headroom for Reddit rate-limit waits.
#
# Script is resumable via disease_ranking_progress.jsonl. --signal tells SLURM
# to send SIGUSR1 to this shell 120s before the time limit hits; the trap below
# catches that, stops the python process, queues the continuation job, and
# exits cleanly — so the job chain keeps going with no manual resubmission.
# If a run finishes all diseases before the warning fires, no signal is sent
# and the chain stops on its own.
#
# Usage: sbatch submit_rank_disease_subs.sh

set -uo pipefail

SCRIPT_PATH="/ncats/users/liky/social_media_data/submit_rank_disease_subs.sh"

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
conda activate reddit

python /ncats/users/liky/social_media_data/rank_disease_subreddits.py &
PY_PID=$!
wait "$PY_PID"
