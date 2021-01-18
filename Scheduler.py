import numpy as np

def cosine_anneal(start, end, ratio):
    '''
    Cosine annealing from start to end as ratio goes from 0.0 to 1.0
    
    Inputs:
    - start : float
    - end : float
    - ratio : float between 0 and 1
    
    Output : 
    - cosine update
    '''
    cos_out = np.cos(np.pi * ratio) + 1
    return end + (start-end)/2 * cos_out

def linear_anneal(start, end, ratio):
    '''
    Linear annealing from start to end as ratio goes from 0.0 to 1.0
    
    Inputs:
    - start : float
    - end : float
    - ratio : float between 0 and 1
    
    Output : 
    - linear update
    '''
    return start + ratio * (end-start)

def exp_anneal(start, end, ratio):
    '''
    Exp annealing from start to end as ratio goes from 0.0 to 1.0
    
    Inputs:
    - start : float
    - end : float
    - ratio : float between 0 and 1
    
    Output : 
    - linear update
    '''
    return start * (end/start) ** ratio


dico = {'linear': linear_anneal, 'cosine': cosine_anneal, 'exp': exp_anneal}


class Scheduler():
    '''
    Used to step from start to end over steps iterations on a schedule defined by func
    '''
    def __init__(self, start, end, num_steps, func):
        '''
        Inputs : 
        - lr_min : minimal learning rate
        - lr_max : maximal learning rate
        - mom_min : minimal momentum
        - mom_max : maximal momentum
        - steps : number iterations

        - func : function
        '''
        
        self.start = start
        self.end = end
        self.n = 0
        self.n_iter = num_steps
        self.function = func
        
    def step(self):
        self.n +=1
        return self.function(self.start, self.end, self.n / self.n_iter)
