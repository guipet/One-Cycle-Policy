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




# Define the Keras model to add callbacks to
def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_dim=784))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model


# In[131]:


# Load example MNIST data and pre-process it
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Limit the data to 1000 samples
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]


# In[132]:


#modele
model = get_model()

#callback
lr_schedule = One_Cycle(0.1, 5, 128)

#fit
model.fit( x_train, y_train, batch_size=128, epochs=5, verbose=1, callbacks=[lr_schedule]);


# In[133]:


lr_schedule.plot()

