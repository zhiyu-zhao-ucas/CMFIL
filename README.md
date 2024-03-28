
# Imitation Learning for Mean Field Games with Correlated Equilibrium

This repository is the official implementation of Imitation Learning for Mean Field Games with Correlated Equilibrium. 


## Requirements

To install requirements:

```setup
conda install --yes --file requirements.txt 
```


## Training

To train CMFIL for tasks (Sequential Squeeze_short, Sequential Squeeze, RPS, Flock, Traffic Flow Prediction), run this command:
```train
python main_{task}.py
```
To train MFIRL:
```train
python MFIRL_{task}.py
```
To train MFAIRL:
```train
python MFAIRL_{task}.py
```


## Evaluation

Result reports and visualization:
```python
import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy


def smooth(data, sm=100):
    smooth_data = deepcopy(data)
    smooth_data = np.convolve(np.ones(sm)/sm, data, 'valid')
    return smooth_data


sns.set(style="whitegrid")
mf01 = torch.load(<dir for result1>)
mf02 = torch.load(<dir for result2>)
mf03 = torch.load(<dir for result>)
e0 = np.vstack((mf01, mf02, mf03))
se = scipy.stats.sem(e0, axis=0)
In = np.arange(smooth(e0.mean(0)).shape[0])
plt.fill_between(In, smooth(e0.mean(0) - se), smooth(e0.mean(0) + se), alpha=0.2)
plt.plot(In, smooth(e0.mean(0)), label=<name>)
print(e0[:,-1].round(5), np.mean(e0[:,-1]).round(5), se[-1].round(5), smooth(e0.mean(0))[-1].round(5))
```



## Contributing

MIT license
