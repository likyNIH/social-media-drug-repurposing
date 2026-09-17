#!/bin/bash
#SBATCH --job-name=build_topic_models
#SBATCH --output=/ncats/users/liky/social_media_data/logs/build_topic_models_%j.out
#SBATCH --error=/ncats/users/liky/social_media_data/logs/build_topic_models_%j.err
#SBATCH --partition=quick_cpu
#SBATCH --account=ncats
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:SIGUSR1@120

# Submits build_topic_models.py as a batch job with real memory headroom.
#
# Running this interactively in the VSCode session OOM-killed partway through
# Narcolepsy (44,742 documents) -- that session itself runs inside a SLURM
# job with only a ~4GB memory cgroup limit, and Top2Vec's doc2vec training
# (vocabulary + embedding matrices) for tens of thousands of often-long
# documents exceeded it. 32G here is a generous margin above the ~3.3GB that
# killed the 4GB job, sized for the largest communities (Narcolepsy,
# visualsnow, CysticFibrosis, each 20,000-45,000 documents).
#
# Resumable: process_subreddit() in build_topic_models.py skips any
# subreddit whose topics.json already exists, so Microtia and
# LongQTSyndrome (already completed during interactive testing) are
# skipped, and a killed/resubmitted run just continues down the list.
#
# Same chain-resubmission safety net as the other batch jobs, in case a
# corpus is larger/slower than expected. Per-disease Top2Vec training took
# well under a minute for the two small communities tested interactively,
# so 4 hours should be ample for all 20, but the trap costs nothing if unused.
#
# Usage: sbatch submit_topic_models.sh

set -uo pipefail

SCRIPT_PATH="/ncats/users/liky/social_media_data/submit_topic_models.sh"

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

python /ncats/users/liky/social_media_data/build_topic_models.py &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Finished cleanly (all communities topic-modeled)."
elif [ "$EXIT_CODE" -eq 137 ]; then
    echo "Killed (likely OOM even at 32G) -- investigate before resubmitting; NOT auto-resubmitting."
    exit "$EXIT_CODE"
else
    echo "Python exited with code $EXIT_CODE -- queuing continuation job..."
    NEXT_JOBID=$(sbatch --parsable "$SCRIPT_PATH")
    echo "Queued continuation job: $NEXT_JOBID"
fi
