timestamp=$(date +%Y%m%d-%H%M%S)
for i in $(seq 5)
do
    echo "Starting" $timestamp "fold $i"
    python train.py --config default_run --timestamp $timestamp --fold $i
done
# python upload_data.py --timestamp $timestamp