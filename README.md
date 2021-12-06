# dirl_pytorch
Pytorch implementation of the [DIRL](https://arxiv.org/abs/2011.07589) model and experiments on digits datasets

original repo [here](https://github.com/ajaytanwani/DIRL)


# Install
Setup virtual environment and install python dependencies:
```
virtualenv python=/usr/bin/python3 env_dirl
source env_dirl/bin/activate
pip install -r requirements.txt
```

# Run experiments
Simply run ``train_digits.py`` to run the MNIST->USPS experiment, or use the -h flag to see other available options
