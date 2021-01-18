# One-Cycle-Policy

The implementation of a One Cycle Policy (OCP) based on [this paper](https://arxiv.org/pdf/1803.09820.pdf).  
OCP is a way to train a model. Instead of keeping a fixed learning rate and a fixed momentum, or even decreasing them over the time, we build two phases:
- phase 1 : we increase the learning rate to a maximal bound while we decrease the momentum to a lower bound.
- phase 2: we decrease the learning rate to a minimal bound and we increase the momentum to a maximal bound.

**Scheduler.py**: It is a class that allow to schedule the learning rate during the phases. There exists three ways to change these parameters during training:
- linearly with the function `linear_anneal`
- with a cosine with the function `cosine_anneal`
- exponentially with the function `exp_anneal` 

**One_Cycle.py**: It is a class based on the *Callback class* of Tensorflow. It uses **Scheduler.py** to schedule the learning rate. It is a callback that allows a model to train with the OCP. At the end of the training, you can plot the evolution of the parameters.

**Be careful**! To get the most out of the OCP, it is important to pay attention to a lot of hyperparameters like the range of the learning rate, the batch size, the weight decay. These hyperparameters vary a lot from a problem to another because of the architecture of the network, the data, the weights...

# Prerequisites
- Tensorflow version 2.5.0 or more recent.
- Numpy version 1.19.4 or more recent.
- Plotly version 4.14.1 or more recent.

# Parameters 
`class One_Cycle(lr_max, epochs = None, batch_size = None, moms = (0.95, 0.85), div_factor = 25., 
                 len_phase1 = 0.3, func = 'cosine', ann = True)`

- `lr_max` : Maximal bound of the learning rate.
- `epochs` : Number of epochs to train the model.
- `batch_size` : Batch size. 
- `moms` : tuple of the maximal and minimal bound of the momentum or beta_1 if it exists. *Default* : (0.95, 0.85).
- `div_factor` : a int that define the lower bound of the learning rate (`lr_max / div_factor`). *Default* : 25.
- `len_phase_1` :  factor between 0 and 1 indicating the percentage of step for phase 1. *Default* : 0.3. 
- `func` : the name of the function to shcedule the parameters. *Default* : 'cosine'
- `ann`: Boolean that informs if we want to anneal the learning rate at the end of the training. *Default* : True

# Result
This is an example of OCP with a cosine evolution
![alttext](https://github.com/guipet/One-Cycle-Policy/blob/main/plots/lr%26mom.png)

There is a small example of comparison between an OCP againt a classical training
![alttext](https://github.com/guipet/One-Cycle-Policy/blob/main/plots/plot_eval.png) 

# References
[1] [A DISCIPLINED APPROACH TO NEURAL NETWORK HYPER-PARAMETERS: PART 1 – LEARNING RATE, BATCH SIZE, MOMENTUM, AND WEIGHT DECAY](https://arxiv.org/pdf/1803.09820.pdf)  
[2] [https://github.com/titu1994/keras-one-cycle](https://github.com/titu1994/keras-one-cycle/tree/06c202996d71491e624ddef53a57858152e93564)  
[3] [https://github.com/titu1994/keras-one-cycle/](https://github.com/titu1994/keras-one-cycle/tree/06c202996d71491e624ddef53a57858152e93564)  
[4] [Super-convergence in Tensorflow 2 with the 1Cycle Policy](https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/)
[5] [The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)
