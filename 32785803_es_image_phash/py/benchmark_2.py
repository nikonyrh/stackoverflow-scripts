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
    
    field_sort = lambda: {'%x' % randint(0, n_samples-1): 'asc'}
    query = {
        'size': 500,
        'sort': [field_sort() for i in range(4)],
        'fields': ['raw']
    }
    #print(json.dumps(query, indent=4, sort_keys=True)); return
    
    docs = [
        doc['fields']['raw'][0]
        for doc in client.search(
            index=index, body=query, request_timeout=1e3
        )['hits']['hits']
    ]
    n_docs = len(docs)
    
    #print('\n'.join(docs[0:10])); return
    
    durations = ([], [])
    matches   = ([], [])
    
    for i in range(n_docs):
        doc = np.array([int(i) for i in docs[i][::-1]]).astype(np.uint64)        
        doc = {'%x' % j:
            int(doc.dot(sampler[:,:,j]).dot(pow2).astype(np.int16))
            for j in range(n_samples)
        }

        #print(json.dumps(doc, indent=2, sort_keys=True)); return

        query = [None, None]

        query[0] = {
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
                'minimum_should_match': 29
            }
        }
        
        query[1] = {
            'fuzzy': {
                'raw': {
                    'value': docs[i],
                    'fuzziness': 2,
                    'max_expansions': 1000000
                }
            }
        }

        #print(json.dumps(query, indent=2)); return
        
        for j in range(len(query)):
            start = time.time()
            n_hits = client.search(index=search_index, body={
                'size': 0,
                'query': query[j]
            }, request_timeout=600)['hits']['total']

            durations[j].append(1000*(time.time() - start))
            matches[j].append(n_hits)

        print('\r%4d/%d %5d %7.2f ms, %5d %7.2f ms' % (i+1, len(docs),
            matches[0][-1], durations[0][-1],
            matches[1][-1], durations[1][-1]
        ), end='')
        sys.stdout.flush()
    
    print('\n\nMean matches: %.3f / %.3f' % (sum(matches[0]) / n_docs, sum(matches[1]) / n_docs))
    print('Mean time:    %.3f / %.3f ms' % (sum(durations[0]) / n_docs, sum(durations[1]) / n_docs))

    durations[0].sort()
    durations[1].sort()
    
    print('\n'.join('  %2dth = %6.2f / %6.2f ms' % (i,
        durations[0][int(round(i*0.01*(n_docs-1)))],
        durations[1][int(round(i*0.01*(n_docs-1)))]
    ) for i in [1, 5, 25, 50, 75, 95, 99]))


if __name__ == '__main__':
    [main(int(sys.argv[i])) for i in range(1, len(sys.argv))]
