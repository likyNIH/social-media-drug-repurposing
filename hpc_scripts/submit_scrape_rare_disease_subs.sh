#!/bin/bash
#SBATCH --job-name=scrape_rare_disease_subs
#SBATCH --output=/ncats/users/liky/social_media_data/logs/scrape_rare_disease_subs_%j.out
#SBATCH --error=/ncats/users/liky/social_media_data/logs/scrape_rare_disease_subs_%j.err
#SBATCH --partition=extended_cpu
#SBATCH --account=ncats
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --signal=B:SIGUSR1@120

# Submits scrape_rare_disease_subreddits.py as a self-resubmitting chain job,
# same pattern as submit_rank_disease_subs.sh.
#
# This job is much bigger than the original scleroderma test case: summed
# across all 20 communities there are 130,000+ posts, each needing its own
# paginated comments fetch. Uses extended_cpu (7-day limit) instead of
# quick_cpu (4-hour limit) to cut down on the number of chain handoffs this
# needs -- extended_cpu's name is inferred from the extended_gpu partition
# confirmed to exist on this cluster (see /ncats/users/liky/testproject/
# test_timecheck.sh); double check it's the right name if sbatch rejects it.
# The chain-resubmission logic is kept as a safety net in case even 7 days
# isn't enough, or the job gets preempted/the node goes down. --signal tells
# SLURM to send SIGUSR1 to this shell 120s before the time limit; the trap
# below stops the current python process, queues the continuation job, and
# exits cleanly.
#
# Resumable at the subreddit and post level: scrape_rare_disease_subreddits.py
# writes each subreddit to a .tmp file (flushed after every post) and only
# renames it to the final path once fully scraped. A mid-subreddit kill
# leaves an incomplete .tmp file that's picked back up from where it left
# off, rather than either looking done (skipped forever) or being redone
# from scratch (confirmed costly: a real crash partway through Narcolepsy's
# ~44,700 posts would otherwise have thrown away tens of thousands of
# already-fetched posts).
#
# Unlike submit_rank_disease_subs.sh, this wrapper also auto-resubmits on an
# actual python crash, not just the graceful time-limit signal -- confirmed
# necessary in production: an unhandled exception (a Unicode edge case in
# clean_text) silently killed a prior run, and since only SIGUSR1 triggered
# resubmission, nothing recovered until a human noticed. A crash after
# running for a while (plausibly a rare data-related edge case, safe to just
# continue from) is distinguished from a crash within the first minute
# (more likely a systemic bug -- code/env issue) by elapsed runtime; the
# latter does NOT auto-resubmit, to avoid silently spinning in a tight
# crash loop if something is fundamentally broken.
#
# Usage: sbatch submit_scrape_rare_disease_subs.sh

set -uo pipefail

SCRIPT_PATH="/ncats/users/liky/social_media_data/submit_scrape_rare_disease_subs.sh"
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
conda activate reddit

START_TIME=$(date +%s)
python /ncats/users/liky/social_media_data/scrape_rare_disease_subreddits.py &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?
ELAPSED=$(( $(date +%s) - START_TIME ))

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Finished cleanly (all communities scraped)."
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
