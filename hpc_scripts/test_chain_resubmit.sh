#!/bin/bash
#SBATCH --job-name=test_chain
#SBATCH --output=/ncats/users/liky/social_media_data/logs/test_chain_%j.out
#SBATCH --error=/ncats/users/liky/social_media_data/logs/test_chain_%j.err
#SBATCH --partition=quick_cpu
#SBATCH --account=ncats
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --signal=B:SIGUSR1@60

# Dry run for the self-resubmission mechanism in submit_rank_disease_subs.sh.
# Uses a 2-minute time limit with the SIGUSR1 warning at T-60s (fires ~60s in)
# and a fake workload (sleep) instead of the real script, so the chain
# mechanics can be verified in ~1 minute instead of ~4 hours.
#
# Usage: sbatch test_chain_resubmit.sh
# Then:  squeue -u $USER          (watch a second job appear)
#        cat logs/test_chain_<jobid>.out   (see the trap fire + resubmit)

set -uo pipefail

SCRIPT_PATH="/ncats/users/liky/social_media_data/test_chain_resubmit.sh"

on_time_limit() {
    echo "[$(date)] SIGUSR1 received — stopping fake workload and queuing continuation job..."
    kill -TERM "$WORK_PID" 2>/dev/null
    wait "$WORK_PID" 2>/dev/null
    NEXT_JOBID=$(sbatch --parsable "$SCRIPT_PATH")
    echo "[$(date)] Queued continuation job: $NEXT_JOBID"
    exit 0
}
trap on_time_limit SIGUSR1

echo "[$(date)] Job $SLURM_JOB_ID starting on $(hostname)"

sleep 600 &
WORK_PID=$!
wait "$WORK_PID"

echo "[$(date)] Fake workload finished normally (should not happen before signal in this test)"
