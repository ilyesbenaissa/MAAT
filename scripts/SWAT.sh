export CUDA_VISIBLE_DEVICES=0
python main.py --anormly_ratio 0.5 --num_epochs 1   --batch_size 256 --win_size 100    --mode train --dataset SWAT  --data_path dataset/SWAT   --input_c 51  --output_c 51
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256 --win_size 100   --mode test    --dataset SWAT   --data_path dataset/SWAT     --input_c 51
