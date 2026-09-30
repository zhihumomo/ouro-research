nohup python -m torch.distributed.run --nproc_per_node=1 run.py > log/train.log 2>&1 &
# python -m torch.distributed.run --nproc_per_node=1 run.py
# tensorboard --logdir=log
# python webui/serve.py --pretrain --port 8901
# python webui/serve.py --port 8901
