python scripts/jax/train_SearchBaselines.py -s 3 -f configs/gp/rbf/inf/np.yaml   -lr 0.002811706625952 --model.kwargs.loss_type ml
python scripts/jax/train_SearchBaselines.py -s 3 -f configs/gp/rbf/inf/anp.yaml  -lr 0.002811706625952 --model.kwargs.loss_type ml

python scripts/jax/train_SearchBaselines.py -s 3 -f configs/gp/rbf/inf/np.yaml   -lr 0.002811706625952 --model.kwargs.loss_type iwae
python scripts/jax/train_SearchBaselines.py -s 3 -f configs/gp/rbf/inf/anp.yaml  -lr 0.002811706625952 --model.kwargs.loss_type iwae

python scripts/jax/train_SearchBaselines.py -s 3 -f configs/gp/rbf/inf/np.yaml   -lr 0.002811706625952 --model.kwargs.loss_type elbo
python scripts/jax/train_SearchBaselines.py -s 3 -f configs/gp/rbf/inf/anp.yaml  -lr 0.002811706625952 --model.kwargs.loss_type elbo
