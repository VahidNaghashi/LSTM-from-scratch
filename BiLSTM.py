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
Wuf = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wff = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wcf = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wof = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wub = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wfb = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wcb = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wob = np.random.randn(hidden_size, hidden_size+vocab_size) * 0.01
Wy = np.random.randn(vocab_size, hidden_size + hidden_size) * 0.01
buf = np.zeros((hidden_size, 1))
bff = np.zeros((hidden_size, 1))
bcf = np.zeros((hidden_size, 1))
bof = np.zeros((hidden_size, 1))
bub = np.zeros((hidden_size, 1))
bfb = np.zeros((hidden_size, 1))
bcb = np.zeros((hidden_size, 1))
bob = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def lossFun(inputs, targets, hprev, cprev, hlast, clast):

    xs, hsf, hsb, csf, csb, usf, usb, fsf, fsb, ccsf, ccsb, osf, osb, ys, ps, concat1, concat2, concat3 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}

    Tx = len(inputs)
    hsf[-1] = np.copy(hprev)
    hsb[Tx] = np.copy(hlast)
    csf[-1] = np.copy(cprev)
    csb[Tx] = np.copy(clast)
    loss = 0

    #Forward pass
    for t in range(Tx):

        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1
        concat1[t] = np.row_stack((hsf[t-1], xs[t]))
        usf[t] = sigmoid(np.dot(Wuf, concat1[t]) + buf)
        fsf[t] = sigmoid(np.dot(Wff, concat1[t]) + bff)
        ccsf[t] = np.tanh(np.dot(Wcf, concat1[t]) + bcf)
        osf[t] = sigmoid(np.dot(Wof, concat1[t]) + bof)

        csf[t] = usf[t]*ccsf[t] + fsf[t]*csf[t-1]
        hsf[t] = osf[t]*np.tanh(csf[t])

    for t in reversed(range(Tx)):

        concat2[t] = np.row_stack((hsb[t+1], xs[t]))
        
        usb[t] = sigmoid(np.dot(Wub, concat2[t]) + bub)
        fsb[t] = sigmoid(np.dot(Wfb, concat2[t]) + bfb)
        ccsb[t] = np.tanh(np.dot(Wcb, concat2[t]) + bcb)
        osb[t] = sigmoid(np.dot(Wob, concat2[t]) + bob)

        csb[t] = usb[t]*ccsb[t] + fsb[t]*csb[t+1]
        hsb[t] = osb[t]*np.tanh(csb[t])
        concat3[t] = np.row_stack((hsf[t], hsb[t]))

        ys[t] = np.dot(Wy,concat3[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        loss += -np.log(ps[t][targets[t]])


    #Back propagation through time
    dWuf = np.zeros_like(Wuf)
    dWff = np.zeros_like(Wff)
    dWcf = np.zeros_like(Wcf)
    dWof = np.zeros_like(Wof)
    dbuf = np.zeros_like(buf)
    dbff = np.zeros_like(bff)
    dbcf = np.zeros_like(bcf)
    dbof = np.zeros_like(bof)
    dWub = np.zeros_like(Wub)
    dWfb = np.zeros_like(Wfb)
    dWcb = np.zeros_like(Wcb)
    dWob = np.zeros_like(Wob)
    dbub = np.zeros_like(bub)
    dbfb = np.zeros_like(bfb)
    dbcb = np.zeros_like(bcb)
    dbob = np.zeros_like(bob)
    dWy = np.zeros_like(Wy)
    dby = np.zeros_like(by)

    dhnextf = np.zeros((hidden_size, 1))
    dcnextf = np.zeros((hidden_size, 1))
    dhnextb = np.zeros((hidden_size, 1))
    dcnextb = np.zeros((hidden_size, 1))
    
    #dxt = {}

    for t in reversed(range(Tx)):

        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
    
        dWy += np.dot(dy, concat3[t].T)
        dby += dy
        dhf = np.dot(Wy[:,:hidden_size].T, dy) + dhnextf
        dcf = dhf * osf[t] * (1 - np.tanh(csf[t])**2) + dcnextf
        
        dusf = dcf * ccsf[t] + dhf * osf[t] * (1-np.tanh(csf[t])**2) * ccsf[t] 
        dfsf = dcf * csf[t-1] + dhf * osf[t] * (1-np.tanh(csf[t])**2) * csf[t-1]
        dccsf = dcf * usf[t] + dhf * osf[t] * (1-np.tanh(csf[t])**2) * usf[t]
        dosf = dhf * np.tanh(csf[t])

        dWuf += np.dot(dusf*usf[t]*(1-usf[t]), concat1[t].T)
        dWff += np.dot(dfsf*fsf[t]*(1-fsf[t]), concat1[t].T)
        dWcf += np.dot(dccsf*(1-ccsf[t]**2), concat1[t].T)
        dWof += np.dot(dosf*osf[t]*(1-osf[t]), concat1[t].T)
        dbuf += dusf*usf[t]*(1-usf[t])
        dbff += dfsf*fsf[t]*(1-fsf[t])
        dbcf += dccsf*(1-ccsf[t]**2)
        dbof += dosf*osf[t]*(1-osf[t])

        dcnextf = dcf*fsf[t] + dhf*osf[t]*(1-np.tanh(csf[t])**2)*fsf[t]
        dhnextf = np.dot(Wuf[:,:hidden_size].T, dusf*usf[t]*(1-usf[t])) + np.dot(Wff[:,:hidden_size].T, dfsf*fsf[t]*(1-fsf[t])) + np.dot(Wcf[:,:hidden_size].T, dccsf*(1-ccsf[t]**2)) + np.dot(Wof[:,:hidden_size].T, dosf*osf[t]*(1-osf[t]))
        #dxt[t] += np.dot(Wuf[:,hidden_size:].T, dusf*usf[t]*(1-usf[t])) + np.dot(Wff[:,hidden_size:].T, dfsf*fsf[t]*(1-fsf[t])) + np.dot(Wcf[:,hidden_size:].T, dccsf*(1-ccsf[t]**2)) + np.dot(Wof[:,hidden_size:].T, dosf*osf[t]*(1-osf[t]))
                                  
    for t in range(Tx):

        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dhb = np.dot(Wy[:,hidden_size:].T, dy) + dhnextb
        dcb = dhb * osb[t] * (1-np.tanh(csb[t])**2) + dcnextb

        dusb = dcb * ccsb[t] + dhb * osb[t] * (1-np.tanh(csb[t])**2) * ccsb[t]
        dfsb = dcb * csb[t+1] + dhb * osb[t] * (1-np.tanh(csb[t])**2) * csb[t+1]
        dccsb = dcb * usb[t] + dhb * osb[t] * (1-np.tanh(csb[t])**2) * usb[t]
        dosb = dhb * np.tanh(csb[t])

        dWub += np.dot(dusb*usb[t]*(1-usb[t]), concat2[t].T)
        dWfb += np.dot(dfsb*fsb[t]*(1-fsb[t]), concat2[t].T)
        dWcb += np.dot(dccsb*(1-ccsb[t]**2), concat2[t].T)
        dWob += np.dot(dosb*osb[t]*(1-osb[t]), concat2[t].T)
        dbub += dusb*usb[t]*(1-usb[t])
        dbfb += dfsb*fsb[t]*(1-fsb[t])
        dbcb += dccsb*(1-ccsb[t]**2)
        dbob += dosb*osb[t]*(1-osb[t])

        dcnextb = dcb*fsb[t] + dhb*osb[t]*(1-np.tanh(csb[t])**2)*fsb[t]
        dhnextb = np.dot(Wub[:,:hidden_size].T, dusb*usb[t]*(1-usb[t])) + np.dot(Wfb[:,:hidden_size].T, dfsb*fsb[t]*(1-fsb[t])) + np.dot(Wcb[:,:hidden_size].T, dccsb*(1-ccsb[t]**2)) + np.dot(Wob[:,:hidden_size].T, dosb*osb[t]*(1-osb[t]))
        #dxt[t] += np.dot(Wub[:,hidden_size:].T, dusb*usb[t]*(1-usb[t])) + np.dot(Wfb[:,hidden_size:].T, dfsb*fsb[t]*(1-fsb[t])) + np.dot(Wcb[:,hidden_size:].T, dccsb*(1-ccsb[t]**2)) + np.dot(Wob[:,hidden_size:].T, dosb*sob[t]*(1-osb[t]))
                                                        
         
     return loss, dWuf, dWff, dWcf, dWof, dbuf, dbff, dbcf, dbof, dWub, dWfb, dWcb, dWob, dbub, dbfb, dbcb, dbob, dWy, dby, hsf[Tx-1], csf[Tx-1], hsb[0], csb[0]
                                  
####################################################
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

######################################################
n, p  = 0, 0
mWuf, mWff, mWcf, mWof, mWub, mWfb, mWcb, mWob, mWy = np.zeros_like(Wuf), np.zeros_like(Wff), np.zeros_like(Wcf), np.zeros_like(Wof), np.zeros_like(Wub), np.zeros_like(Wfb), np.zeros_like(Wcb), np.zeros_like(Wob), np.zeros_like(Wy)
mbuf, mbff, mbcf, mbof, mbub, mbfb, mbcb, mbob, mby = np.zeros_like(buf), np.zeros_like(bff), np.zeros_like(bcf), np.zeros_like(bof), np.zeros_like(bub), np.zeros_like(bfb), np.zeros_like(bcb), np.zeros_like(bob), np.zeros_like(by)

smooth_loss = -np.log(1.0 / vocab_size) * seq_length

iterations = 1000
for it in range(iterations):

    #prepare inputs
    if p+seq_length+1 >= len(data) or n==0:

        hprevf = np.zeros((hidden_size, 1))
        cprevf = np.zeros((hidden_size, 1))
        hlastb = np.zeros((hidden_size, 1))
        clastb = np.zeros((hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    #if n%100 == 0:
        #sample_ix = sample(hprev, inputs[0], 200)
       # txt = ''.join([ix_to_char[ix] for ix in sample_ix])
       # print( '----\n %s \n----' %(txt, ))
        


    #forward seq in network and get gradients
    loss, dWuf, dWff, dWcf, dWof, dbuf, dbff, dbcf, dbof, dWub, dWfb, dWcb, dWob, dbub, dbfb, dbcb, dbob, dWy, dby, hprevf, cprevf, hlastb, clastb = lossFun(inputs, targets, hprevf, cprevf, hlastb, clastb)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n%100 == 0:
        print ('iter %d, loss: %f' % (n, smooth_loss))


    #preform parameter update with adagrad
    for param, dparam, mem in zip([Wuf, Wff, Wcf, Wof, buf, bff, bcf, bof, Wub, Wfb, Wcb, Wob, bub, bfb, bcb, bob, Wy, by],
                                  [dWuf, dWff, dWcf, dWof, dbuf, dbff, dbcf, dbof, dWub, dWfb, dWcb, dWob, dbub, dbfb, dbcb, dbob, dWy, dby],
                                  [mWuf, mWff, mWcf, mWof, mbuf, mbff, mbcf, mbof, mWub, mWfb, mWcb, mWob, mbub, mbfb, mbcb, mbob, mWy, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    #move data pointer
    p += seq_length
    #increase iteration counter
    n += 1
                                  
        
        
    
