DEPTH_DATA_URL="https://www.dropbox.com/s/qtab28cauzalqi7/depth_data.tar.gz?dl=1"
DATA_EXTRACT_DIR="./data"

PRETRAINED_URL="https://www.dropbox.com/s/356r36lfpyzhcht/pretrained_models.tar.gz?dl=1"
PRETRAINED_EXTRACT_DIR="./"

wget -c $DEPTH_DATA_URL -O - | tar -xz -C $DATA_EXTRACT_DIR

mkdir $PRETRAINED_DIR
wget -c $PRETRAINED_URL -O - | tar -xz -C $PRETRAINED_EXTRACT_DIR
