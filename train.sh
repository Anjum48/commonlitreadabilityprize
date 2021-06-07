config=${1:-default_run}

slug=$(python -c "from coolname import generate_slug; print(generate_slug(3))")

for seed in 48 123 2021
do
    timestamp=$(date +%Y%m%d-%H%M%S)
    for i in $(seq 5)
    do
        echo "Starting" $timestamp "fold $i"
        python train.py --config $config --timestamp $timestamp --fold $i --seed $seed --slug $slug
    done
    python agg_scores.py
    python infer.py --timestamp $timestamp --seed $seed
    # python upload_data.py --timestamp $timestamp
done