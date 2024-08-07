# This repository is for the paper "Robust and Faster Zeroth-order Minimax Optimization: Complexity and Applications".



Black-box  poisoning  attack  against  logistic  regression  model
---------------------------------------------------

Download the `epsilon_test` dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon. Execute the following Matlab code for reprocessing:

```
[epsilony, epsilonx] = libsvmread('your data path');
epsilonx = full(epsilonx);
mmean = mean(epsilonx);
sstd = std(epsilonx);
epsilonx = (epsilonx - mmean)./sstd;
samples = epsilonx;
labels = epsilony;
save('epsilon_mean_std.mat', 'samples', 'labels');
```

run

```
python Main_poison_attack_ZO_GDEGA.py
```

