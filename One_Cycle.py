import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback, Callback
from tensorflow.keras import backend as K
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class One_Cycle(Callback):
    '''
    Train a NN with the One Cycle Policy. The LR will be in two phases:
    - phase 1 : we increase the LR from lr_min to lr_max
    - phase 2 : we decrease the LR from lr_max to final_lr
    
    There are 3 ways to update the LR : cosine, linear and exp. If the optimizer of the model has a momentum 
    (or beta_1 in the same way), it will update the momentum inversely to the learning rate. We can modify the 
    lenght of the phase 1 and choose if wa want to anneal to 0 the LR at the end.
    
    From Leslie Smith's paper (https://arxiv.org/pdf/1803.09820.pdf)
    '''
    
    def __init__(self, lr_max, epochs = None, batch_size = None, moms = (0.95, 0.85), div_factor = 25., 
                 len_phase1 = 0.3, func = 'cosine', ann = True):
        '''        
        Inputs :
        - lr_max : Float 
        - moms : Tuple (mom_max, mom_min)
        - div_factor : Float. Factor by which we divide lr_max (to obtain lr_min and final_lr)
        - len_phase_1 : Float between 0 and 1 indicating the percentage of step for phase 1 (ascending phase)
        - func : String. Function used to schedule the parameters
        - ann : Boolean. If we anneal the LR at the end 
        '''
        
        super(One_Cycle, self).__init__()
        
        #we instantiate lr_min, mom_max and mom_min and _moms (a boolean for the final print)
        self.lr_min   = lr_max / div_factor
        self.mom_max  = moms[0] 
        self.mom_min  = moms[1]
        self._moms = True

        
        if batch_size is None:
            raise ValueError("must provide a value to batche_size")
        
        if epochs is None: 
            raise ValueError("must provide a value to epochs")
        
        if dico.get(func, None) is None:
            raise ValueError(str(func) + " doesn't exist. Choose between linear, cosine or exp.")
            
        if len_phase1 > 1 or len_phase1 < 0:
            raise ValueError("len_phase1 must be between 0 and 1 excluded")
        
        if not isinstance(ann, bool):
            raise ValueError("ann must be a boolean")
            
            
        #we instantiate the final value of the LR, the function of the scheduler, the number of iterations 
        #of phase 1 and phase 2.
        self.final_lr = lr_max / (div_factor*1e4) if ann else self.lr_min
        function = dico[func]
        num_steps = np.ceil(len(x_train)/batch_size) * epochs
        self.steps_phase1 = num_steps * len_phase1
        self.steps_phase2 = num_steps - self.steps_phase1
        
        #on instancie les phases
        self.phases = [[Scheduler(self.lr_min, lr_max, self.steps_phase1, function), Scheduler(self.mom_max, self.mom_min, self.steps_phase1, function)], 
                 [Scheduler(lr_max, self.final_lr, self.steps_phase2, function), Scheduler(self.mom_min, self.mom_max, self.steps_phase2, function)]]
    
        
        
    def on_train_begin(self, logs = None):
        '''
        Instancie le début de l'entrainement
        '''
        self.reset()
        
        K.set_value(self.model.optimizer.learning_rate, self.lr_min)
        self.set_momentum(self.mom_max)
        
    
    def on_train_batch_begin(self, batch, logs = None):
        '''
        update self.lrs et self.moms
        '''
        self.lrs.append( K.get_value(self.model.optimizer.lr) )
        self.moms.append( self.get_momentum() )

        
    def on_train_batch_end(self, batch, logs = None):
        '''
        Update le LR, le momentum et la phase.
        '''
        #update la step
        self.step += 1
        if self.step > self.steps_phase1:
            self.phase = 1
            
        #on update le learning rate et le momentum
        new_lr, new_mom = self.update_parameters()
            
        #affecte le nouveau lr et mom au modèle
        K.set_value(self.model.optimizer.learning_rate, new_lr)         
        self.set_momentum(new_mom)
        
        
    def reset(self):
        '''
        reset les paramètres pour un nouvel entrainement
        
        - self.step : l'itération actuelle
        - self.phase : l'une des phases du OneCycle
        - self.lrs : liste contenant les différents learning rate
        - self.moms : liste contenant les différents momentum 
        '''
        self.step = 0
        self.phase = 0
        self.lrs = []
        self.moms = []
        
    def update_parameters(self):
        '''
        Cosine update le momentum et le lr
                
        Output :
        - lr : float, learning rate
        - mom : float, momentum
        '''
        lr = self.phases[self.phase][0].step()
        mom = self.phases[self.phase][1].step()
        
        return lr, mom

    def get_momentum(self):
        '''
        Get the value of the momemtum or beta_1 of the optimizer if it exists
        '''
        
        #momentum
        try:
            mom = K.get_value(self.model.optimizer.momentum)
            self.mom_name = "momentum"
            return mom
        except AttributeError:
            pass
        
        #beta_1
        try : 
            beta_1 = K.get_value(self.model.optimizer.beta_1)
            self.mom_name = 'beta_1'
            
            return beta_1
        except AttributeError:
            self._moms = False
            pass
            
    def set_momentum(self, new_mom):
        '''
        Modify the value of the momentum in the model
        '''
        
        #momentum
        try:
            return K.set_value(self.model.optimizer.momentum, new_mom)
        except AttributeError:
            pass
        
        #beta_1
        try:
            return K.set_value(self.model.optimizer.beta_1, new_mom)
        except AttributeError:
            pass
            
        
        
    def plot(self):
        
        if self._moms :
            fig = make_subplots(rows=1, cols=2)       
            fig.add_trace(go.Scatter(y=self.lrs, mode="lines", name = "Learning rate"), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.moms, mode="lines", name = self.mom_name), row=1, col=2)
            fig.update_xaxes(title_text="Training iterations")       
            fig.show()
            
        else :
            fig = go.Figure(data=go.Scatter(y=self.lrs, mode = "lines"))
            fig.update_xaxes(title_text="Training iterations") 
            fig.update_yaxes(title_text="Learning rate") 
            fig.show()


