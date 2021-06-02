config=${1:-default_run}

for seed in 48 123 2021
do
    timestamp=$(date +%Y%m%d-%H%M%S)
    for i in $(seq 5)
    do
        echo "Starting" $timestamp "fold $i"
        python train.py --config $config --timestamp $timestamp --fold $i --seed $seed
    done
    # python upload_data.py --timestamp $timestamp
done