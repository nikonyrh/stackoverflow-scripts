from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from random import randint
import json
import time
import sys

def main():
    index = 'image_hashes'
    client = Elasticsearch('localhost:9200')
    
    field_sort = lambda: {'%x' % randint(0, 1023): 'asc'}
    docs = [
        doc['_source']
        for doc in client.search(index='image_hashes', body={
            'size': 500,
            'sort': [field_sort() for i in range(4)]
        })['hits']['hits']
    ]
    
    #print(json.dumps(docs[0], indent=4)); return
    
    durations = []
    matches   = []
    
    for i in range(len(docs)):
        start = time.time()
        doc   = docs[i]
        
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
                    for field in doc.keys() if field != 'raw'
                ],
                'minimum_should_match': 100
            }
        }
        
        #print(json.dumps(query, indent=2)); return
        
        n_hits = client.search(index='image_hashes*', body={
            'size': 0,
            'query': query
        })['hits']['total']
        
        durations.append(1000*(time.time() - start))
        matches.append(n_hits)
        
        print('%4d %5d %6.2f ms' % (i, n_hits, durations[-1]))
        sys.stdout.flush()
    
    print('\nMean time:    %.3f ms' % (sum(durations) / len(docs)))
    print('Mean matches: %.2f' % (sum(matches) / len(matches)))

    durations.sort()
    print('\n'.join('  %2dth = %.2fms' % (i, durations[round(i*0.01*(len(durations)-1))])
        for i in [1, 5, 25, 50, 75, 95, 99]))


if __name__ == '__main__':
    main()
