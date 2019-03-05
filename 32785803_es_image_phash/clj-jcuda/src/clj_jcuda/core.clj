(ns clj-jcuda.core
  (:require [uncomplicate.commons.core :as commons.core]
            [uncomplicate.clojurecuda.core :as cc.core :refer [with-context context]]            
            [clojure.repl :refer [source doc]]
            [clojure.pprint :refer [pprint]]
            
            [nikonyrh-utilities-clj.core :as u]))


(set! *warn-on-reflection* true)

(defn read-numbers [fname n]
  (let [; Well this seems a bit stupid way of converting a 64-bit unsigned int to a signed one :o
        max-long (apply * (repeat 63 2N))
        to-long #(long (if (>= % max-long) (- max-long %) %))]
    (with-open [rdr (clojure.java.io/reader fname)]
      (->> rdr
           line-seq
           (take n)
           (map #(-> % bigint to-long))
           long-array))))


(def hashes-cpu
  ; Refer to py/generate_to_file.py on how to generate this file
  (read-numbers "/home/wrecked/projects/stackoverflow-scripts/32785803_es_image_phash/py/numbers.txt" 64))

(def n-elems (count hashes-cpu))
(def matches-cpu (int-array n-elems))

(cc.core/init)
(def gpu (cc.core/device 0))
(def ctx (context gpu))

(comment
  (commons.core/release ctx))


(cc.core/in-context ctx
  (def hashes-gpu  (cc.core/mem-alloc (* 8 n-elems)))
  (def matches-gpu (cc.core/mem-alloc (* 4 n-elems)))
  (cc.core/memcpy-host! hashes-cpu hashes-gpu))


(cc.core/in-context ctx
  (def kernel-fn
    (-> "
      typedef unsigned long long int uint64;
      
      extern \"C\" __device__ int hammingWeight32(int v) {
        v = v - ((v>>1) & 0x55555555);
        v = (v & 0x33333333) + ((v>>2) & 0x33333333);
        
        return ((v + (v>>4) & 0xF0F0F0F) * 0x1010101) >> 24;
      }
      
      extern \"C\" __device__ int hammingDistance64(const uint64 a, const uint64 b) {
        const uint64 delta = a ^ b;
        return hammingWeight32(delta & 0xffffffffULL) + hammingWeight32(delta >> 32);
      }
      
      extern \"C\" __global__ void myFinder(const int n, const uint64 *hashes, const uint64 target, int *results) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < n) {
          results[i] = hammingDistance64(hashes[i], target);
        }
      }
    " cc.core/program cc.core/compile! cc.core/module (cc.core/function "myFinder"))))


(cc.core/in-context ctx
  (cc.core/launch! kernel-fn (cc.core/grid-1d n-elems 32) (cc.core/parameters n-elems hashes-gpu (nth hashes-cpu 3) matches-gpu))
  (cc.core/memcpy-host! matches-gpu matches-cpu)
  
  (seq matches-cpu))






