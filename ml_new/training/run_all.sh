#!/bin/zsh
# Chain: wait for pitch to finish → delete PopBuTFy pyin NPZs → train breath → train onset
cd /Users/uddhavjain/Desktop/Drexel/VocalStars/VocalStars
source ml/.venv/bin/activate

PITCH_LOG=ml_new/checkpoints/pitch_allpyin/train_out.log
POPBUTFY_DIR=ml_new/data/extracted_pyin_popbutfy

echo "Waiting for pitch training to complete (watching $PITCH_LOG)..."
until grep -q "Training complete" "$PITCH_LOG" 2>/dev/null; do
  sleep 60
done
echo "Pitch training done."
tail -3 "$PITCH_LOG"

echo "Freeing disk: deleting $POPBUTFY_DIR ..."
rm -rf "$POPBUTFY_DIR"
echo "Freed. Disk now:"
df -h /Users/uddhavjain

echo "Starting breath model training..."
python ml_new/training/train_breath.py \
  --manifest ml_new/data/extracted_pyin/manifest.csv \
  --output-dir ml_new/checkpoints/breath \
  --epochs 30 --batch-size 64 2>&1

echo "Starting onset model training..."
python ml_new/training/train_onset.py \
  --manifest ml_new/data/extracted_pyin/manifest.csv \
  --output-dir ml_new/checkpoints/onset \
  --epochs 30 --batch-size 64 2>&1

echo "All three models trained."
echo "Disk usage:"
du -sh ml_new/checkpoints/pitch_allpyin/best.pt ml_new/checkpoints/breath/best.pt ml_new/checkpoints/onset/best.pt 2>/dev/null
