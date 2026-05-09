source activate_environment.sh
cargo run --bin bin_deepmath_pipeline -- \
    --dataset-name "gsm8k" \
 --num-samples 200 \
 --model gpt