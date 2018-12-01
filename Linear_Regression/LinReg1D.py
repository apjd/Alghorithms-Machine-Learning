#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import sympy as s
import matplotlib.pyplot as plt


# In[101]:


class TraningData:
    '''Defines the structures of training data'''
    def __init__(self, x, y):
        '''x: array containing x-axis data, y: array containing y-axis data'''
        self.x = x
        self.y = y
        

class LinReg1D:
    '''Class containing functions needed for '''
    
    def __init__(self, TrainingData):
        self.TD = TrainingData
        
        # Initialize current values for a and b to zero
        self.a = 0
        self.b = 0
        
        c = 0
        while c < 1000:
            print('Count: %d' % c)
            self.gradient_descent(step=0.1)
            c = c + 1
        
    
    def hypothesis(cls, x, a, b):
        '''Hypothesis function for the linear regression is a line: y = slope*x+offset'''
        return a*x+b
    
    def mean_squared_error(cls, a, b):
        '''Return the current mean squared error based on current "self.a" and "self.b". In linear regression, the mean squared error is used as cost function'''
        
        h = cls.hypothesis(TD.x, a, b)
            
        
        return (1/(2*np.size(TD.y))) * np.sum(np.square(h - TD.y))
    
    def gradient_descent(cls, step):
        '''Update "cls.a" and "cls.b" based on gradient descent minimization procedure'''
        
        a, b = s.symbols('a b')
        a_new = cls.a - step*s.diff(cls.mean_squared_error(a, cls.b), a).subs(a, cls.a)
        b_new = cls.b - step*s.diff(cls.mean_squared_error(cls.a, b), b).subs(b, cls.b)
        
        cls.a = a_new
        cls.b = b_new


# In[102]:


TD = TraningData(x=np.array([0, 1, 2, 3, 4, 5])+np.random.random(6), y=np.array([0, 1, 2, 3, 4, 5])+np.random.random(6))


# In[103]:


LR = LinReg1D(TD)


# In[105]:


plt.plot(TD.x, LR.a*TD.x+LR.b, TD.x, TD.y)


# In[ ]:




