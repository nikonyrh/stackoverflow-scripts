import numpy as np
from hashlib import sha1
import random
import json
import time
import sys

dim_in   = 32
dim_rand = 25  # Number of "clusters" is about 2^(dim_in - dim_rand)
dim_out  = 64

n_samples  = 64
b_p_sample = 16

rand_scale = 1e-2

# Make projections deterministic
np.random.seed(seed=123)
proj = np.random.randn(dim_in, dim_out)
proj[0:dim_rand,:] *= rand_scale

# Make generated document unique
np.random.seed(seed=int(time.time()))

def get_sampler(n_samples, bits_per_sample):
	result = np.zeros((dim_out,bits_per_sample,n_samples), dtype=np.uint64)
	hash   = lambda i, j: sha1(
		('%d-%d' % (i, j)).encode('utf-8')
	).hexdigest()
	
	ind2 = list(range(bits_per_sample))
	
	for i in range(n_samples):
		ind1 = [(j, hash(i, j)) for j in range(dim_out)]
		ind1.sort(key=lambda j: j[1])
		ind1 = sorted([j[0] for j in ind1[0:bits_per_sample]])
		
		result[ind1,ind2,i] = 1;
	
	# Ah damn, maybe this should have been in a reverse order :P
	# But I am too lazy to re-generate all data to ES
	pow2 = 2**np.arange(bits_per_sample).astype(np.uint64)
	return result, pow2


def main():
	n_out   = int(100e6)
	n_batch = 10000
	
	sampler, pow2 = get_sampler(n_samples, b_p_sample)
	
	with open('numbers.txt', 'w') as f:
		for i in range(n_out // n_batch):
			data = np.random.randn(n_batch, dim_in)
			hash = (data.dot(proj) > 0).astype(np.uint64)
			hash_int = hash.dot(2**np.arange(dim_out).astype(np.uint64))
			
			f.write('\n'.join('%d' % j for j in hash_int) + '\n')


if __name__ == '__main__':
	main()