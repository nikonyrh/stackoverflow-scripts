import numpy as np

n_bits = 64
m_values = list(range(1, 48+1))

format = lambda arr: ' ' + (' '.join(
    '%12.9f' % i if not isinstance(i, str) else i
    for i in arr
)).replace('.', ',')

P_dm = {}

#print(format([np.nan] + m_values))

for dist in [float(d) for d in range(16+1)]:
    row = [dist]
    
    P_dm[int(dist)] = {}
    
    for m in m_values:
        p = 1 - np.prod([(n_bits - dist - i) / (n_bits - i) for i in range(m)])
        P_dm[dist][m] = p
        row.append(p)
    
    #print(format(row))


#import json; print(json.dumps(P_dm, indent=4, sort_keys=True))

n_trial = int(2**10)
s_len   = int(2**4)

_lf_cache    = {}
_lf_cache[0] = 0

def log_factorial(n):
    if n not in _lf_cache:
        for i in range(1, n+1):
            if i not in _lf_cache:
                _lf_cache[i] = np.log(i) + (0 if i == 1 else _lf_cache[i-1])
    
    return _lf_cache[n]


def log_nCk(n, k):
    return log_factorial(n) - (log_factorial(k) + log_factorial(n - k))


def log(f, margin=1e-20):
    return np.log(max(margin, min(1 - margin, f)))


if False:
    P = []
    for i in range(n_bits+1):
        P.append(np.exp(log_nCk(n_bits, i)))
    
    total = sum(P)
    for i in range(n_bits+1):
        print(('%3d %.30f' % (i, P[i]/total)).replace('.', ','))
    
    import sys; sys.exit(0)


d_arr = sorted(P_dm.keys())
print(format([np.nan] + ['D' + str(i) for i in d_arr]))

cumusm = {i: 0 for i in d_arr}
prev = -1

for k in range(0, n_trial+1):
    row = [n_trial - k]
    
    for dist in d_arr:
        p = 1 - P_dm[dist][s_len]
        
        a = log_nCk(n_trial, k)
        b = (n_trial - k) * log(p)
        c = k * log(1 - p)
        #print(repr([a, b, c]))
        
        cumusm[dist] += np.exp(a + b + c)
        row.append(cumusm[dist])
    
    i = np.floor(250*(float(k) / n_trial)**8)
    
    if i != prev:
        print(format(row))
        prev = i
