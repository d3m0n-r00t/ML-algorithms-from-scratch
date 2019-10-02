import numpy as np 

class rnn():
    def step(self,x):
        #hidden layer
        self.h=np.tanh(np.dot(self.W_hh,self.h)+np.dot(self.W_xh,x))
        #output layer
        y=np.dot(self.W_hy,self.h)
        '''
        3 matrices. W_hh,W_xh,W_hy
        np.dot --> matrix multiplication with two inputs previous hidden state and present input
        The two intermediates interact with addition. Then it is sqashed by tanh
        '''
        return y

#Single hidden layer. Or single step
'''
nn = rnn()
y=nn.step(x)
'''
#2-layer network
'''
y1=rnn.step(x)
y=rnn.step(y1)
'''