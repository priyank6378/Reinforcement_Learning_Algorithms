import numpy as np


def e_greedy(q_vals, e=0.1) : 
    '''
    This function chooses the action greedily with probability 1-e
    and randomly with probability e.
    '''
    if np.random.random() < e : 
        return np.random.randint(len(q_vals))
    else : 
        return np.argmax(q_vals)
    
def softmax_policy(q_vals , tau=1) : 
    '''
    This function chooses the action according to the softmax policy.
    '''
    prob = np.exp(q_vals/tau)/np.sum(np.exp(q_vals/tau))
    return np.random.choice(len(q_vals), p=prob)