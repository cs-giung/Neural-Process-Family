python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/mnist/banp.yaml  -lr 0.000500000000000 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/mnist/canp.yaml  -lr 0.000500000000000 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/mnist/anp.yaml   -lr 0.000500000000000 --datasets.train.batch_size 128 --train.num_epochs 10 --model.kwargs.loss_type ml
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/mnist/bnp.yaml   -lr 0.000500000000000 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/mnist/cnp.yaml   -lr 0.000500000000000 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/mnist/np.yaml    -lr 0.000500000000000 --datasets.train.batch_size 128 --train.num_epochs 10 --model.kwargs.loss_type ml
