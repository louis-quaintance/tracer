#!/bin/bash
# Provision a GPU instance on AWS, upload data, train, download model, terminate.
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - A region with g4dn instances (default: eu-west-2 / London)
#
# Usage:
#   bash aws_setup.sh
#
# This script will:
#   1. Create a key pair (if needed)
#   2. Create a security group (if needed)
#   3. Find the Deep Learning AMI automatically
#   4. Launch a g4dn.xlarge (~$0.50/hr)
#   5. Wait for it to be ready
#   6. Upload data + training script
#   7. Install deps + train
#   8. Download the trained model
#   9. Terminate the instance

set -e

REGION="${AWS_DEFAULT_REGION:-eu-west-2}"
INSTANCE_TYPE="g4dn.xlarge"
KEY_NAME="golfball-train2"
KEY_FILE="golfball-train2.pem"
SG_NAME="golfball-train-sg"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE"
echo ""

# ── Step 1: Key pair ──
if [ ! -f "$PROJECT_DIR/$KEY_FILE" ]; then
    echo "── Creating key pair ──"
    aws ec2 create-key-pair \
        --region "$REGION" \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text > "$PROJECT_DIR/$KEY_FILE"
    chmod 400 "$PROJECT_DIR/$KEY_FILE"
    echo "  Saved: $KEY_FILE"
else
    echo "── Key pair exists: $KEY_FILE ──"
fi

# ── Step 2: Security group ──
echo "── Setting up security group ──"
SG_ID=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --group-names "$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "NONE")

if [ "$SG_ID" = "NONE" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Golf ball model training" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0
fi
echo "  Security group: $SG_ID"

# ── Step 3: Find Deep Learning AMI ──
echo "── Finding Deep Learning AMI ──"
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
        "Name=state,Values=available" \
        "Name=architecture,Values=x86_64" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "  DL AMI not found, trying alternative search..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=*Deep Learning*PyTorch*Ubuntu*22.04*" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "ERROR: Could not find a Deep Learning AMI in $REGION"
    echo "Try setting a different region: AWS_DEFAULT_REGION=us-east-1 bash aws_setup.sh"
    exit 1
fi
echo "  AMI: $AMI_ID"

# ── Step 4: Launch instance ──
echo "── Launching $INSTANCE_TYPE ──"
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=golfball-training}]' \
    --query 'Instances[0].InstanceId' --output text)

echo "  Instance: $INSTANCE_ID"
echo "$INSTANCE_ID" > "$PROJECT_DIR/.aws_instance_id"

echo "  Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "  Public IP: $PUBLIC_IP"

# ── Step 5: Wait for SSH ──
echo "── Waiting for SSH to be ready ──"
for i in $(seq 1 30); do
    if ssh -i "$PROJECT_DIR/$KEY_FILE" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@"$PUBLIC_IP" "echo ready" 2>/dev/null; then
        break
    fi
    echo "  Attempt $i/30..."
    sleep 10
done

# ── Step 6: Upload data + scripts ──
echo "── Uploading data and training script ──"
scp -i "$PROJECT_DIR/$KEY_FILE" -o StrictHostKeyChecking=no \
    "$PROJECT_DIR/train.py" ubuntu@"$PUBLIC_IP":~/

scp -i "$PROJECT_DIR/$KEY_FILE" -o StrictHostKeyChecking=no -r \
    "$PROJECT_DIR/data" ubuntu@"$PUBLIC_IP":~/

echo "  Upload complete"

# ── Step 7: Install deps + train (in tmux so it survives disconnects) ──
echo "── Installing dependencies and starting training ──"
ssh -i "$PROJECT_DIR/$KEY_FILE" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" << 'REMOTE'
set -e
echo "Installing ultralytics..."
pip3 install ultralytics --quiet

echo "Starting training in tmux session 'train'..."
tmux new-session -d -s train "python3 train.py --epochs 100 --imgsz 416 --batch 32 2>&1 | tee train.log; echo DONE > /tmp/train_done"
REMOTE

echo "  Training started in tmux. Polling for completion..."
while true; do
    DONE=$(ssh -i "$PROJECT_DIR/$KEY_FILE" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" \
        "cat /tmp/train_done 2>/dev/null || echo NO")
    if [ "$DONE" = "DONE" ]; then
        echo "  Training complete!"
        break
    fi
    # Show last line of log
    ssh -i "$PROJECT_DIR/$KEY_FILE" -o StrictHostKeyChecking=no ubuntu@"$PUBLIC_IP" \
        "tail -1 train.log 2>/dev/null" || true
    sleep 30
done

# ── Step 8: Download model ──
echo "── Downloading trained model ──"
scp -i "$PROJECT_DIR/$KEY_FILE" -o StrictHostKeyChecking=no \
    ubuntu@"$PUBLIC_IP":~/runs/detect/golfball/weights/best.pt \
    "$PROJECT_DIR/best.pt"

echo "  Model saved to: $PROJECT_DIR/best.pt"

# ── Step 9: Terminate ──
echo "── Terminating instance ──"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" > /dev/null
echo "  Instance $INSTANCE_ID terminated"

# Clean up security group and key pair (optional, commented out)
# aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID"
# aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME"
# rm "$PROJECT_DIR/$KEY_FILE"

echo ""
echo "══════════════════════════════════════"
echo "  Done! Model saved to: best.pt"
echo ""
echo "  Run the tracker:"
echo "    python golf_tracer.py your_video.mp4 --model best.pt --show"
echo "══════════════════════════════════════"
