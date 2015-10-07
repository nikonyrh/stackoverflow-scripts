from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
import json
import time
import sys
from math import ceil, log
import random


def rand(p, n=1, min_val=1):
    # Mean and standard deviation are expected to be "p"
    #return [max(1, round(-p*log(random.random()))) for i in range(n)]
    
    return [int(round(
        min_val + p / 1.147 * ((1 - random.random())**-0.2 - 1)
    )) for i in range(n)]



def main(argv):
    index = 'user_topics'
    client = Elasticsearch('localhost:9200')
    index_client = IndicesClient(client)
    
    if index_client.exists(index):
        index_client.delete(index)
    
    index_client.create(index=index, body={
        'settings': {
            'number_of_shards':   4,
            'number_of_replicas': 0
        },
        'mappings': {
            'user': {
                'properties': {
                    #'id': {
                    #    'type': 'long',
                    #    'doc_values': True
                    #},
                    'topics': {
                        'type': 'integer',
                        'doc_values': True
                    },
                    'n_topics': {
                        'type': 'integer',
                        'doc_values': True
                    }
                }
            }
        }
    })
    
    n_users           = int(argv[1])
    n_topics          = int(argv[2]) * 0.15
    n_topics_per_user = int(argv[3]) * 4.2
    
    docs_per_chunk = int(2e4)
    n_chunks       = int(ceil(n_users / docs_per_chunk))
    
    start_time = time.time()
    
    for i_chunk in range(1, n_chunks+1):
        docs = []
        
        for i in range(docs_per_chunk):
            n_user_topics = rand(n_topics_per_user)[0]
            topics = list(set(rand(n_topics, n_user_topics)))
            
            doc_id = str(random.getrandbits(63))

            docs.append('{"index":{"_index": "%s", "_type": "user", "_id": "%s"}})' % (index, doc_id))
            docs.append(json.dumps({
                #'id':      doc_id,
                'topics':   topics,
                'n_topics': len(topics)
            }))
        
        #print(json.dumps(json.loads(docs[1]), indent=4)); return
        
        try:
            response = client.bulk(body='\n'.join(docs))
        except:
            # Even when an exception is thrown typically documents were stored in ES
            sleep_seconds = 10
            print('\rHTTP timed out, sleeping %d seconds...' % sleep_seconds)
            time.sleep(sleep_seconds)
        
        print('\rChunk %5d/%d, %5.2f%%' % (i_chunk, n_chunks, i_chunk*100.0/n_chunks), end='')
    
    index_time = time.time()
    print('\nCalling optimize, indexing took %.1f s...' % (index_time - start_time))
    sys.stdout.flush()
    
    index_client.optimize(index=index, max_num_segments=3, request_timeout=1e6)
    print('Optimization done in %.1f s' % (time.time() - index_time))

if __name__ == '__main__':
    main(sys.argv if len(sys.argv) > 1 else [
        None, 20e6, 20e3 , 10
    ])
