import numpy as np
from numpy.linalg import norm
####################################### UTILS #######################################

def risk(K, x, xstar):
    return (x-xstar).transpose() @ K @ (x-xstar) / 2


def clip(x,c):
    if len(x.shape) > 1:
        raise Exception("You need to clip a matrix")
    k = np.minimum(1, c / norm(x))
    return k * x

def clip_matrix(x,c):
    return np.minimum(1, c / norm(x, axis = 1))[:,None] * x


####################################### Vanilla SGD #######################################

def vanilla_ode(K,T, noise_std, x0, xstar, lr):
    dt = 0.05
    d = len(x0)

    vals, vecs = np.linalg.eigh(K)
    v = ((x0-xstar) @ vecs)**2 / 2
    R = np.dot(vals,v)
    risks = []

    ode_time = []
    iters = int(T / dt)
    for i in range(iters + 1):
        t = i * dt
        R = np.dot(vals,v)
        
        update = -lr * 2 * v * vals + lr**2 * (vals * (2*R+ noise_std**2)) / (2 * d)        
        
        v = v + dt * update

        ode_time.append(t)
        risks.append(R)
    return np.array(risks), np.array(ode_time)

def clipped_ode(K,T, x0, xstar, lr, mu, nu, mu_nu_args):
    dt = 0.1
    d = np.trace(K)

    vals, vecs = np.linalg.eigh(K)
    v = ((x0-xstar) @ vecs)**2 / 2
    R = np.dot(vals,v)
    risks = []

    ode_time = []
    iters = int(T / dt)
    for i in range(iters + 1):
        t = i * dt
        R = np.dot(vals,v)
        
        
        update = -lr * 2 * v * vals* mu(R,**mu_nu_args) + lr**2 * (vals * nu(R,**mu_nu_args)) / (2* d)
        
        v = v + dt * update

        ode_time.append(t)
        risks.append(R)
    return np.array(risks), np.array(ode_time), vals, vecs


def one_pass_clipped_sgd(K, A, y, x, target, lrk, ck):
    r = []
    times = []

    for i,(a,b) in enumerate(zip(A,y)):
        if i % 20 == 0:
            times.append(i)
            r.append(risk(K,x,target))

        grad = (np.dot(x,a) - b) * a

        if ck > 0:
            grad = clip(grad, ck)

        x = x - lrk * grad

    times.append(i+1)
    r.append(risk(K,x,target))
    return np.array(r), np.array(times)

def clipped_hsgd(K, sqrtK, beta,T,x):    
    """ DO NOT USE """
    dt = 0.01
    sqrtdt = np.sqrt(dt)
    N = int(T/dt)

    risks = []
    time = []

    for i in range(N):
        R = risk(K,x,beta)
        risks.append(R)
        time.append(dt * i)

        delta_B = np.random.randn(ambient_d) * sqrtdt

        gradP = K @ (x-beta)

        # x = x - gamma * gradP * H_STU(R,scale,df,c)*dt + gamma * np.sqrt(G_STU(R,scale,df,c) * (2*R + eta**2)/d) * sqrtK @ delta_B
        x = x - gamma * gradP * H_STU(R,scale,df,c)*dt + gamma * np.sqrt(G_STU(R,scale,df,c) /d) * sqrtK @ delta_B


    time.append(dt * N)
    risks.append(risk(K,x,beta))

    time = np.array(time)
    risks = np.array(risks)
    return risks, time


def one_pass_sgd(K, data, targets, x0, xstar, lrk):
    x = x0
    risk_vals = []
    times = []

    for i,(a,b) in enumerate(zip(data,targets)):
        if i % 20 == 0:
            times.append(i)
            risk_vals.append(risk(K, x,xstar))

        grad = (np.dot(x,a) - b) * a        
        signed_grad = grad

        x = x - lrk * signed_grad

    times.append(i+1)
    risk_vals.append(risk(K, x,xstar))
    
    return np.array(risk_vals), np.array(times)

####################################### Sign SGD #######################################

def one_pass_sign_sgd(K, data, targets, x0, xstar, lrk):
    x = x0
    risk_vals = []
    times = []

    for i,(a,b) in enumerate(zip(data,targets)):
        if i % 20 == 0:
            times.append(i)
            risk_vals.append(risk(K, x,xstar))

        grad = (np.dot(x,a) - b) * a        
        signed_grad = np.sign(grad)

        x = x - lrk * signed_grad

    times.append(i+1)
    risk_vals.append(risk(K, x,xstar))
    
    return np.array(risk_vals), np.array(times)


def sign_ode(K, Kbar, Ktilde, T, noise_std, x0, xstar, lr):
    print('Precomputations...')
    U, S, Vt = np.linalg.svd(Kbar)

    risks = []

    dt = 0.001
    d = len(x0)
    
    KtildeK = Ktilde @ K
    y = np.array([np.inner(x0-xstar,Vt[j,:]) * np.inner(x0-xstar, K @ U[:,j]) for j in range(d)]) / 2
    var_force = np.array([np.inner(Vt[j,:], KtildeK @ U[:,j]) for j in range(d)])
    

    ode_time = []
    iters = int(T / dt)

    # print(S)
    # print('*' * 10)
    # print(var_force)    

    print('Iterating...')
    for i in range(iters):
        t = i * dt
        R = np.sum(y)
        P = R + noise_std**2/2

        
        update = -lr * 4 * y * S / (np.pi * np.sqrt(2 * P)) + lr**2 * var_force / (2 * d)

        y = y + dt * update

        ode_time.append(t)
        risks.append(R)
    return np.array(risks), ode_time



def sign_hsgd(K, Kbar, sqrt_Ktilde, x0, xstar, T):    
    dt = 0.005
    sqrtdt = np.sqrt(dt)    
    N = int(T/dt)

    risks = []
    time = []
    x = x0

    for i in range(N):
        time.append(dt * i)
        delta_B = np.random.randn(d) * sqrtdt

        mean = - 2 * lr * Kbar @ (x-xstar) / np.sqrt(2 * risk(K,x,xstar) + noise_std**2) / np.pi

        x = x + mean * dt + lr * sqrt_Ktilde @ delta_B / np.sqrt(d)
        
        risks.append(risk(K,x,xstar))

    time = np.array(time)
    risks = np.array(risks)

    return risks, time
