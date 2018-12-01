"""
Minimal character-level LSTM

"""

import numpy as np
#data I/O
data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

#hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

#model parameters
Wf = np.random.randn(hidden_size, hidden_size+vocab_size)*0.01
Wu = np.random.randn(hidden_size, hidden_size+vocab_size)*0.01
Wc = np.random.randn(hidden_size, hidden_size+vocab_size)*0.01
Wo = np.random.randn(hidden_size, hidden_size+vocab_size)*0.01
Wy = np.random.randn(vocab_size, hidden_size)*0.01
bf = np.zeros((hidden_size, 1))
bu = np.zeros((hidden_size, 1))
bc = np.zeros((hidden_size, 1))
bo = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def lossFun(inputs, targets, hprev, cprev):

    """
    Inputs, targets are both list of integers
    """
    xs, hs, cs, ys, ps, fs, us, os, ccs, concat = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)
    loss = 0
    #Forward pass of LSTM
    for t in range(len(inputs)):

        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        #concat[t][ :hidden_size] = hs[t-1]
        concat[t] = np.row_stack((hs[t-1], xs[t]))
        #concat[t][hidden_size: ] = xs[t]
        fs[t] = sigmoid(np.dot(Wf, concat[t]) + bf)
        us[t] = sigmoid(np.dot(Wu, concat[t]) +  bu)
        os[t] = sigmoid(np.dot(Wo, concat[t]) + bo)
        ccs[t] = np.tanh(np.dot(Wc, concat[t]) + bc)
        cs[t] = fs[t]*cs[t-1] + us[t]*ccs[t]
        hs[t] = os[t]*np.tanh(cs[t])
        ys[t] = np.dot(Wy, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        loss += -np.log(ps[t][targets[t]])

    #Backward pass of LSTM
    dWf, dWu, dWc, dWo, dWy = np.zeros_like(Wf), np.zeros_like(Wu), np.zeros_like(Wc), np.zeros_like(Wo), np.zeros_like(Wy)
    dbf, dbu, dbc, dbo, dby = np.zeros_like(bf), np.zeros_like(bu), np.zeros_like(bc), np.zeros_like(bo), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    for t in reversed(range(len(inputs))):

        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Wy.T, dy) + dhnext 
        dc = os[t]*(1-np.tanh(cs[t])**2)*dh + dcnext
        #dh = dh + dhnext
        do = np.tanh(cs[t])*dh * os[t] * (1-os[t])
        df = (dc*cs[t-1] + os[t]*(1-np.tanh(cs[t])**2)*cs[t-1]*dh) * (fs[t]*(1-fs[t]))
        du = (ccs[t]*dc + os[t]*(1-np.tanh(cs[t])**2)*ccs[t]*dh) * (us[t]*(1-us[t]))
        dcc = (us[t]*dc + os[t]*(1-np.tanh(cs[t])**2)*us[t]*dh) * (1 - ccs[t]*ccs[t])
        dhnext = np.dot(Wf[:,:hidden_size].T, df) + np.dot(Wu[:,:hidden_size].T, du) + np.dot(Wo[:,:hidden_size].T, do) + np.dot(Wc[:,:hidden_size].T, dcc)
        dcnext = dc*fs[t] + os[t]*(1-np.tanh(cs[t])**2)*fs[t]*dh
        dx = np.dot(Wf[:,hidden_size:].T,df) + np.dot(Wu[:,hidden_size:].T,du) + np.dot(Wo[:,hidden_size:].T,do) + np.dot(Wc[:,hidden_size:].T,dcc)
        dWf += np.dot(df, concat[t].T)
        dWu += np.dot(du, concat[t].T)
        dWc += np.dot(dcc, concat[t].T)
        dWo += np.dot(do, concat[t].T)
        dbf += df
        dbu += du
        dbc += dcc
        dbo += do

    for dparam in [dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby, hs[len(inputs)-1], cs[len(inputs)-1]
        

def sample(h, c, seed_ix, n):

    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    concat = np.zeros((hidden_size+vocab_size, 1))
    
    for t in range(n):

        concat[:hidden_size] = h
        concat[hidden_size:] = x
        fs = sigmoid(np.dot(Wf, concat) + bf)
        us = sigmoid(np.dot(Wu, concat) +  bu)
        os = sigmoid(np.dot(Wo, concat) + bo)
        ccs = np.tanh(np.dot(Wc, concat) + bc)
        c = fs*c + us*ccs
        h = os*np.tanh(c)
        ys = np.dot(Wy, h) + by
        p = np.exp(ys) / np.sum(np.exp(ys)) 
        
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

    return ixes


def sigmoid(x):

    return 1 / (1 + np.exp(-x))

######################################################
n, p  = 0, 0
mWf, mWu, mWc, mWo, mWy = np.zeros_like(Wf), np.zeros_like(Wu), np.zeros_like(Wc), np.zeros_like(Wo), np.zeros_like(Wy)
mbf, mbu, mbc, mbo, mby = np.zeros_like(bf), np.zeros_like(bu), np.zeros_like(bc), np.zeros_like(bo), np.zeros_like(by)

smooth_loss = -np.log(1.0 / vocab_size) * seq_length

iterations = 10000
for it in range(iterations):

    #prepare inputs
    if p+seq_length+1 >= len(data) or n==0:

        hprev = np.zeros((hidden_size, 1))
        cprev = np.zeros((hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    if n%100 == 0:
        sample_ix = sample(hprev, cprev, inputs[0], 200)
        txt = ''.join([ix_to_char[ix] for ix in sample_ix])
        print( '----\n %s \n----' %(txt, ))
        


    #forward seq in network and get gradients
    loss, dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n%100 == 0:
        print ('iter %d, loss: %f' % (n, smooth_loss))


    #preform parameter update with adagrad
    for param, dparam, mem in zip([Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by],
                                  [dWf, dWu, dWc, dWo, dWy, dbf, dbu, dbc, dbo, dby],
                                  [mWf, mWu, mWc, mWo, mWy, mbf, mbu, mbc, mbo, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    #move data pointer
    p += seq_length
    #increase iteration counter
    n += 1
        
