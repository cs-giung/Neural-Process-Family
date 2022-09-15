# python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/celeba/banp.yaml  -lr 0.000889139705019 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/celeba/canp.yaml  -lr 0.000889139705019 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/celeba/anp.yaml   -lr 0.000889139705019 --datasets.train.batch_size 128 --train.num_epochs 10 --model.kwargs.loss_type ml
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/celeba/bnp.yaml   -lr 0.000889139705019 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/celeba/cnp.yaml   -lr 0.000889139705019 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/celeba/np.yaml    -lr 0.000889139705019 --datasets.train.batch_size 128 --train.num_epochs 10 --model.kwargs.loss_type ml
