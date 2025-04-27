CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m fastchat.serve.model_worker \
    --port 31021 --worker http://localhost:31021 \
    --host localhost \
    --model-names your-model-name \
    --model-path /model/path \
    --max-gpu-memory 31Gib \
    --dtype float16 \
    --num-gpus 8