from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
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


def main(index_num):
    n_out      = int(10e6)
    n_batch    = int(4e3)
    n_batches  = n_out // n_batch
    index      = 'image_hashes_%02d' % index_num
    
    client = Elasticsearch('localhost:9200')
    index_client = IndicesClient(client)
    
    if index_client.exists(index):
        print('Not deleting %s!' % index); return; sys.exit(1)
        index_client.delete(index)
    
    es_short = {
        'type': 'short',
    }
    
    field_name = lambda i: '%x' % i
    fields = {field_name(i): es_short for i in range(n_samples)}
    fields['raw'] = {
        'type': 'string',
        'store': True,
        'index': 'not_analyzed',
        'doc_values': True
    }
    
    index_client.create(index=index, body={
        'settings': {
            'number_of_shards':   4,
            'number_of_replicas': 0
        },
        'mappings': {
            'images': {
                '_source': {'enabled': False},
                'properties': fields
            }
        }
    })
    
    sampler, pow2 = get_sampler(n_samples, b_p_sample)
    start_time = time.time()
    
    for i_batch in range(1, n_batches+1):
        data = np.random.randn(n_batch, dim_in)
        hash = (data.dot(proj) > 0).astype(np.uint64)
        hash_int = hash.dot(2**np.arange(dim_out).astype(np.uint64))
		
        #print('\n'.join(repr(i.astype(np.uint8)) for i in hash)); return
        
        sampled = np.vstack(
            hash.dot(sampler[:,:,j]).dot(pow2)
            for j in range(n_samples)
        ).astype(np.int16).T.tolist()
        
        #print(repr(sampled)); print(repr([len(sampled), len(sampled[0])])); return
        
        docs = []
        
        for i in range(n_batch):
            doc = {
                field_name(j): sampled[i][j] for j in range(n_samples)
            }
            doc['raw'] = '{0:064b}'.format(hash_int[i])
            doc_id = random.getrandbits(63)
            
            docs.append('{"index":{"_index": "%s", "_type": "images", "_id": "%d"}})' % (index, doc_id))
            docs.append(json.dumps(doc))
        
        #print(json.dumps(json.loads(docs[1]), indent=4)); return
        
        try:
            response = client.bulk(body='\n'.join(docs))
        except:
            # Even when an exception is thrown typically documents were stored in ES
            sleep_seconds = 10
            print('\rHTTP timed out, sleeping %d seconds...' % sleep_seconds)
            time.sleep(sleep_seconds)

        print('\rChunk %5d/%d, %5.2f%%' % (i_batch, n_batches, i_batch*100.0/n_batches), end='')
    
    index_time = time.time()
    print('\nCalling optimize, indexing took %.1f s...' % (index_time - start_time))
    sys.stdout.flush()
    
    index_client.optimize(index=index, max_num_segments=3, request_timeout=1e6)
    print('Optimization done in %.1f s' % (time.time() - index_time))


if __name__ == '__main__':
    for i in range(1, 10+1):
        main(i)
