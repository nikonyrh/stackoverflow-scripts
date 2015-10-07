from elasticsearch import Elasticsearch
import elasticsearch.helpers
from time import time
import json
import sys
import os

index = 'user_topics'
client = Elasticsearch('localhost:9200', timeout=int(1e6))


def get_stats():
    fname = 'stats.json'
    result = None
    
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            result = json.load(f)
        
        doc_id = result['doc']['hits']['hits'][0]['_id']
        
        if not client.exists(index=index, doc_type='user', id=doc_id):
            # New data has been generated, must re-query stats
            result = None
    
    if not result:
        result = {
            'aggs': client.search(
                index=index,
                body={
                  "size": 0,
                  "aggs": {
                    "topic_percentiles": {
                      "percentiles": {
                        "field": "n_topics",
                        "percents": [
                          5,
                          25,
                          50,
                          75,
                          95,
                          99,
                          99.9
                        ]
                      }
                    },
                    "topic_stats": {
                      "extended_stats": { "field": "n_topics" }
                    },
                    "distinct_topics": {
                      "cardinality": {
                        "field": "topics",
                        "precision_threshold": 100000
                      }
                    }
                  }
                }
            ),
            'doc': client.search(index=index, body={"size": 1})
        }
        
        with open(fname, 'w') as f:
            f.write(json.dumps(result, indent=2, sort_keys=True))
    
    return result['aggs']


def get_users(topics, count=False):
    query = {
      "_source": False,
      "query": {
        "filtered": {
          "filter": {
            "terms": {
              "topics": topics,
              "execution": "plain",
              "_cache": False
            }
          }
        }
      }
    }
    
    if count:
        query['size'] = 0;
        return [client.search(index=index, body=query)]
    
    results = elasticsearch.helpers.scan(
        client,
        index=index,
        scroll='1m',
        query=query
    )
    
    # Generator in Python 3.x
    return (int(result['_id']) for result in results)


def main(argv):
    stats = get_stats()
    
    n_users  = 20
    n_topics = 3
    do_count = False
    
    user_topics = [
        hit['_source']['topics'][0:n_topics] for hit in client.search(
            index=index,
            body={
                "size": n_users,
                "query": {
                    "filtered": {
                        "filter": {
                            "range": { "n_topics": {"gte": n_topics} }
                        }
                    }
                }
            }
        )['hits']['hits']]
    
    #print(json.dumps(user_topics, indent=2)); return
    
    durations = []
    count_n_durations = []
    
    for topics in user_topics:
        start = time()
        user_ids = []
        
        for user_id in get_users(topics, count=do_count):
            user_ids.append(user_id)
            
            #if len(user_ids) == 10000: break
        
        durations.append(time() - start)
        
        count = user_ids[0]['hits']['total'] if do_count else len(user_ids)
        count_n_durations.append((count, durations[-1]))
        
        print('Topics %s, users %d, time %.3f s' % (
            repr(topics), count, durations[-1]
        ))
        sys.stdout.flush()
    
    durations = sorted(durations)
    print('\nmin/mean/max: %.3f / %.5f / %.3f s\n' % (
        durations[0], sum(durations) / len(durations), durations[-1]
    ))
    
    for cnd in count_n_durations:
        print('%d %.4f' % cnd)


if __name__ == '__main__':
    main(sys.argv)

