set -ex
python train.py --gpu_ids 0,1 --dataroot ./datasets/facades_p --name facades_pix2pix_res --model pix2pix_res --direction BtoA --lambda_L1 100 --dataset_mode aligned