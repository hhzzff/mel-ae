if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
fi
export CUDA_VISIBLE_DEVICES=2

DEST_DIR="exp/train-libritts-251113-2010-no_layernm_downupsample-overlapdownupsample-kl5.0/"
CONFIG_FILE="conf/24khzVAEkl5.yml"
MELAE_PATH="model/MelVAE.py"
DISCRIMINATOR_PATH="model/discriminator.py"

if [ -d "$DEST_DIR" ]; then
    echo "Error: Directory '$DEST_DIR' already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p $DEST_DIR
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$DEST_DIR/"
cp "$CONFIG_FILE" "$DEST_DIR/"
cp "$MELAE_PATH" "$DEST_DIR/"
cp "$DISCRIMINATOR_PATH" "$DEST_DIR/"
echo "Copied trainVAE.sh, $CONFIG_FILE, MelVAE.py, and discriminator.py to $DEST_DIR"

python bin/trainVAE.py --args.load "$CONFIG_FILE" --save_path "$DEST_DIR"

# if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
#     export PYTHONPATH="$PYTHONPATH:$(pwd)"
# fi
# export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nproc_per_node gpu bin/train.py --args.load conf/24khz.yml --save_path exp/train-libritts-251112-1641/