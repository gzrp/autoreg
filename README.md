# autoreg

.gitconfig
```
[user]
	name = gzhangruipeng@gmail.com
	email = gzhangruipeng

[http]
	proxy = http://10.62.181.38:7890

[https]
	proxy = http://10.62.181.38.7890

```

echo 'export PYTHONPATH="/home/yourname/mylibs:$PYTHONPATH"' >> ~/.bashrc

```angular2html
export http_proxy="http://10.62.181.38:7890"
export https_proxy="http://10.62.181.38:7890"
```

创建环境
```bash
conda create -n py312 python=3.12
conda activate py312
pip3 install torch torchvision

import torch
torch.cuda.is_available()


pip install -U "ray[data,train,tune,serve]"
pip install scikit-learn
pip install timm
pip install matplotlib
pip install hpbandster ConfigSpace

pip install nvitop


```

export PYTHONPATH="/data/ruipeng/workdir/autoreg:$PYTHONPATH"



### 实验一

# 随机
``` 
python exp_random.py \
    --dataset adult \
    --batch_size 64 \
    --seed 42 \
    --device cuda \
    --num_cpus 2 \
    --num_gpus 1 \
    --max_concurrent_trials 1 \
    --lr 1e-3 \
    --momentum 0.9 \
    --max_epochs 4 \
    --num_samples 4 \
    --trail_num_cpus 2 \
    --trail_num_gpus 1 \
    --trail_metric bacc \
    --trail_mode max \
    --exp_name random \
    --storage ~/ray_results
```

hyperband
```
python exp_hyperband.py \
    --dataset adult \
    --batch_size 64 \
    --seed 42 \
    --device cuda \
    --num_cpus 4 \
    --num_gpus 1 \
    --max_concurrent_trials 4 \
    --lr 1e-3 \
    --momentum 0.9 \
    --max_epochs 4 \
    --num_samples 10 \
    --trail_num_cpus 1 \
    --trail_num_gpus 0.1 \
    --trail_metric bacc \
    --trail_mode max \
    --exp_name hyperband \
    --storage ~/ray_results \
    --reduction_factor 2
```


bohb
```
python exp_bohb.py \
    --dataset adult \
    --batch_size 64 \
    --seed 42 \
    --device cuda \
    --num_cpus 4 \
    --num_gpus 1 \
    --max_concurrent_trials 4 \
    --lr 1e-3 \
    --momentum 0.9 \
    --max_epochs 4 \
    --num_samples 10 \
    --trail_num_cpus 1 \
    --trail_num_gpus 0.1 \
    --trail_metric bacc \
    --trail_mode max \
    --exp_name bohb \
    --storage ~/ray_results \
    --reduction_factor 2
```


asha
```
python exp_asha.py \
    --dataset adult \
    --batch_size 64 \
    --seed 42 \
    --device cuda \
    --num_cpus 4 \
    --num_gpus 1 \
    --max_concurrent_trials 4 \
    --lr 1e-3 \
    --momentum 0.9 \
    --max_epochs 4 \
    --num_samples 10 \
    --trail_num_cpus 1 \
    --trail_num_gpus 0.1 \
    --trail_metric bacc \
    --trail_mode max \
    --exp_name asha \
    --storage ~/ray_results \
    --reduction_factor 2
```

age
```
python exp_agevo.py \
    --dataset adult \
    --batch_size 64 \
    --seed 42 \
    --device cuda \
    --num_cpus 4 \
    --num_gpus 1 \
    --max_concurrent_trials 4 \
    --lr 1e-3 \
    --momentum 0.9 \
    --max_epochs 4 \
    --num_samples 10 \
    --trail_num_cpus 1 \
    --trail_num_gpus 0.1 \
    --trail_metric bacc \
    --trail_mode max \
    --exp_name agevo \
    --storage ~/ray_results \
    --reduction_factor 2 \
    --population_size 10 \
    --sample_size 3
```

2-phase
```
python exp_2phase.py \
    --dataset adult \
    --batch_size 64 \
    --seed 42 \
    --device cuda \
    --num_cpus 4 \
    --num_gpus 1 \
    --max_concurrent_trials 4 \
    --lr 1e-3 \
    --momentum 0.9 \
    --max_epochs 4 \
    --num_samples 10 \
    --trail_num_cpus 1 \
    --trail_num_gpus 0.1 \
    --trail_metric bacc \
    --trail_mode max \
    --exp_name exp_2phase \
    --storage ~/ray_results \
    --reduction_factor 2 \
    --population_size 10 \
    --sample_size 3 \
    --k_n 0.2 \
    --max_step 300
```



