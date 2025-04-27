CUDA_VISIBLE_DEVICES=0 nohup python main.py run=pruning_final index=2.3 > 2.3_fast_0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python main.py run=pruning_final index=2.3 > 2.3_fast_1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python main.py run=pruning_final index=2.3 > 2.3_fast_2.txt &

CUDA_VISIBLE_DEVICES=0 nohup python main.py run=pruning_final index=2.8 > 2.8_fast_0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python main.py run=pruning_final index=2.8 > 2.8_fast_1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python main.py run=pruning_final index=2.8 > 2.8_fast_2.txt &