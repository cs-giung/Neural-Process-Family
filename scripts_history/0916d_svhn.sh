python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/svhn/banp.yaml  -lr 0.001581138830084 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/svhn/canp.yaml  -lr 0.001581138830084 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/svhn/anp.yaml   -lr 0.001581138830084 --datasets.train.batch_size 128 --train.num_epochs 10 --model.kwargs.loss_type ml
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/svhn/bnp.yaml   -lr 0.001581138830084 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/svhn/cnp.yaml   -lr 0.001581138830084 --datasets.train.batch_size 128 --train.num_epochs 10
python scripts/jax/train_SearchBaselines_image.py -s 3 -f configs/image/svhn/np.yaml    -lr 0.001581138830084 --datasets.train.batch_size 128 --train.num_epochs 10 --model.kwargs.loss_type ml
