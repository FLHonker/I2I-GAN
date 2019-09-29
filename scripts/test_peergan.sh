set -ex
python test.py --dataroot ./datasets/maps_p --name maps_peer --model peer_gan --direction AtoB --dataset_mode aligned --norm batch
