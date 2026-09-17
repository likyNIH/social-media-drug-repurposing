#!/bin/bash
#SBATCH --job-name=label_topics
#SBATCH --partition=normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --output=/ncats/users/liky/social_media_data/logs/label_topics_%j.out
#SBATCH --error=/ncats/users/liky/social_media_data/logs/label_topics_%j.err

# Same GPU/model config as scleroderma_drug_extraction.sh (gemma3-27b needs
# the same 4 GPUs to load regardless of workload size), but this job is
# much lighter: ~915 topics total, each a short keyword list + a few short
# excerpts, vs. 4,598+ full posts per community in the extraction step.
# Expect this to finish well within a normal_gpu session.
#
# Resumable at the subreddit level: label_topics.py skips any community
# whose topic_labels.json already exists.
#
# Usage: sbatch submit_label_topics.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate test-llama

python /ncats/users/liky/social_media_data/label_topics.py
