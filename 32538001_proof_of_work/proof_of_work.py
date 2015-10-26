from multiprocessing import Pool
from hashlib import sha256
from time import time


def find_solution(args):
    salt, nBytes, nonce_range = args
    target = '0' * nBytes
    
    for nonce in xrange(nonce_range[0], nonce_range[1]):
        result = sha256(salt + str(nonce)).hexdigest()
        
        #print('%s %s vs %s' % (result, result[:nBytes], target)); sleep(0.1)
        
        if result[:nBytes] == target:
            return (nonce, result)
    
    return None


def proof_of_work(salt, nBytes):
    n_processes = 8
    batch_size = int(2.5e5)
    pool = Pool(n_processes)
    
    nonce = 0
    
    while True:
        nonce_ranges = [
            (nonce + i * batch_size, nonce + (i+1) * batch_size)
            for i in range(n_processes)
        ]
        
        params = [
            (salt, nBytes, nonce_range) for nonce_range in nonce_ranges
        ]
        
        # Single-process search:
        #solutions = map(find_solution, params)
        
        # Multi-process search:
        solutions = pool.map(find_solution, params)
        
        print('Searched %d to %d' % (nonce_ranges[0][0], nonce_ranges[-1][1]-1))
        
        # Find non-None results
        solutions = filter(None, solutions)
        
        if solutions:
            return solutions
        
        nonce += n_processes * batch_size
        

if __name__ == '__main__':
    start = time()
    solutions = proof_of_work('abc', 6)
    print('\n'.join('%d => %s' % s for s in solutions))
    print('Solution found in %.3f seconds' % (time() - start))

