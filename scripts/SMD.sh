export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 1   --batch_size 128 --win_size 100  --mode train --dataset SMD  --data_path dataset/SMD   --input_c 38
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 128 --win_size 100 --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38  
