source activate_environment.sh
until nc -z localhost 8000; do
  echo "Waiting for endpoint..."
  sleep 2
done

cargo run --bin bin_deepmath_pipeline -- \
    --dataset-name "gsm8k" \
 --num-samples 200 \
 --model qwen