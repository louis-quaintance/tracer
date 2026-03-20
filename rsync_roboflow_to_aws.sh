#!/usr/bin/env bash
#
# Rsync roboflow_data to an AWS EC2 instance.
#
# Usage:
#   ./rsync_roboflow_to_aws.sh <user@host> [remote_path]
#
# Examples:
#   ./rsync_roboflow_to_aws.sh ubuntu@ec2-12-34-56-78.compute-1.amazonaws.com
#   ./rsync_roboflow_to_aws.sh ubuntu@ec2-12-34-56-78.compute-1.amazonaws.com /home/ubuntu/tracer/
#
# Requires: ssh key access to the remote host (use -i flag via RSYNC_SSH if needed).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [remote_path]"
    echo ""
    echo "  user@host    SSH destination (e.g. ubuntu@1.2.3.4)"
    echo "  remote_path  Remote directory (default: ~/tracer/)"
    exit 1
fi

REMOTE_HOST="$1"
REMOTE_PATH="${2:-~/tracer/}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DATA="$SCRIPT_DIR/roboflow_data"

if [ ! -d "$LOCAL_DATA" ]; then
    echo "ERROR: $LOCAL_DATA does not exist"
    exit 1
fi

echo "Syncing roboflow_data to ${REMOTE_HOST}:${REMOTE_PATH}"
echo "  Local:  $LOCAL_DATA"
echo "  Remote: ${REMOTE_HOST}:${REMOTE_PATH}roboflow_data/"
echo ""

# Create remote directory if it doesn't exist
ssh "$REMOTE_HOST" "mkdir -p ${REMOTE_PATH}"

rsync -avz --progress \
    "$LOCAL_DATA/" \
    "${REMOTE_HOST}:${REMOTE_PATH}roboflow_data/"

# Copy training script
rsync -avz --progress \
    "$SCRIPT_DIR/train_roboflow.py" \
    "${REMOTE_HOST}:${REMOTE_PATH}train_roboflow.py"

echo ""
echo "Done. Data synced to ${REMOTE_HOST}:${REMOTE_PATH}roboflow_data/"
echo ""
echo "To train on the remote machine:"
echo "  ssh ${REMOTE_HOST}"
echo "  cd ${REMOTE_PATH}"
echo "  python train_roboflow.py --epochs 100 --batch 8"
