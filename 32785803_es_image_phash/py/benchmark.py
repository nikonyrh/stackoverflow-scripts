from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from random import randint
import numpy as np
import json
import time
import sys

from generate import get_sampler, n_samples, b_p_sample


sampler, pow2 = get_sampler(n_samples, b_p_sample)


def main(n_indexes):
    client = Elasticsearch('localhost:9200')
    index_client = IndicesClient(client)

    index = 'image_hashes_01'

    if n_indexes < 1:
        search_index = 'image_hashes*'
    else:
        search_index = ','.join('image_hashes_%02d' % i for i in range(2,n_indexes+2))

        aliases = set(index_client.get_alias(index='image_hashes_*').keys())
        for tmp in search_index.split(','):
            if tmp not in aliases:
                search_index = 'image_hashes*'
                break

    print('Searching from ' + search_index)
    
    field_sort = lambda: {'%x' % randint(0, 1023): 'asc'}
    query = {
        'size': 500,
        'sort': [field_sort() for i in range(4)],
        'fields': ['raw']
    }
    #print(json.dumps(query, indent=4, sort_keys=True)); return
    
    docs = [
        np.array([int(i) for i in doc['fields']['raw'][0][::-1]]).astype(np.uint64)
        for doc in client.search(
            index=index, body=query, request_timeout=1e3
        )['hits']['hits']
    ]
    
    #print(repr(docs[0])); return
    
    durations = []
    matches   = []
    
    for i in range(len(docs)):
        start = time.time()
        doc   = docs[i]
        
        doc = {'%x' % j:
            int(doc.dot(sampler[:,:,j]).dot(pow2).astype(np.int16))
            for j in range(n_samples)
        }

        #print(json.dumps(doc, indent=2, sort_keys=True)); return

        query = {
            'bool': {
                'should': [
                    {
                    'filtered': {
                        'filter': {
                                'term': {
                                    field: doc[field],
                                    '_cache': False
                                }
                            }
                        }
                    }
                    for field in sorted(doc.keys())
                ],
                'minimum_should_match': 300
            }
        }
        
        #print(json.dumps(query, indent=2)); return
        
        n_hits = client.search(index=search_index, body={
            'size': 0,
            'query': query
        }, request_timeout=600)['hits']['total']
        
        durations.append(1000*(time.time() - start))
        matches.append(n_hits)
        
        print('\r%4d/%d %5d %7.2f ms' % (i+1, len(docs), n_hits, durations[-1]), end='')
        sys.stdout.flush()
    
    print('\n\nMean matches: %.2f' % (sum(matches) / len(matches)))
    print('Mean time:    %.3f ms' % (sum(durations) / len(docs)))

    durations.sort()
    print('\n'.join('  %2dth = %6.2f ms' % (i, durations[int(round(i*0.01*(len(durations)-1)))])
        for i in [1, 5, 25, 50, 75, 95, 99]))


if __name__ == '__main__':
    [main(int(sys.argv[i])) for i in range(1, len(sys.argv))]
