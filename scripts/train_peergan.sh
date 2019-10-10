set -ex
python train.py --gpu_ids 0,1 --dataroot ./datasets/maps_p --name maps_peer --model peer_gan --direction AtoB --lambda_L1 100 --dataset_mode aligned