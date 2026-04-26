## Privacy-Compatible Authentication in Federated Learning via Reversible Watermarking and TEE

This paper is submitted to ESORICS 2026 

### Environment

Our code does not rely on special libraries or tools, so it can be easily integrated with most environment settings. 

### Dataset

CIFAR-10 and CIFAR-100 datasets are available on `torchvision` and will be downloaded automatically.

Tiny-ImageNet can be easily downloaded from Kaggle.

### Example

Generally, to run a case with default settings, you can easily use the following command:

```
python federated.py \
--poison_frac 0.3 --num_corrupt 4 \
--aggr alignins --data cifar10 --attack badnet
```

If you want to run a case with non-IID settings, you can easily use the following command:

```
python federated.py \
--poison_frac 0.3 --num_corrupt 4 \
--non_iid --alpha 0.5 \
--aggr alignins --data cifar10 --attack badnet
```

Here,

| Argument        | Type       | Description   | Choice |
|-----------------|------------|---------------|--------|
| `aggr`         | str   | Defense method applied by the server | avg, alignins, rlr, mkrum, mmetric, lockdown, foolsgold, rfa|
| `data`    |   str     | Main task data        | cifar10, cifar100, tinyimagenet |
| `num_agents`         | int | Number of clients in FL   | N/A |
| `attack`         | str | Attack method   | badnet, DBA, neurotoxin, pgd |
| `poison_frac`         | float | Data poisoning ratio   | [0.0, 1.0] |
| `num_corrupt`         | int | Number of malicious clients in FL   | [0, num_agents//2-1] |
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A |
| `beta`         | float | Data heterogeneous degree     | [0.1, 1.0]|

For other arguments, you can check the `federated.py` file where the detailed explanation is presented.


## Acknowledgment
Our code is constructed on https://github.com/JiiahaoXU/AlignIns#, big thanks to their contribution!
