set -ex
python train.py --gpu_ids 0,1 --dataroot ./datasets/maps_p --name maps_pix2pix_res --model pix2pix_res --direction AtoB --lambda_L1 100 --dataset_mode aligned