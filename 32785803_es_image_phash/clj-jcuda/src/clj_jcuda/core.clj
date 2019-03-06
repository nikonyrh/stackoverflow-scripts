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
  (read-numbers "/home/wrecked/projects/stackoverflow-scripts/32785803_es_image_phash/py/numbers.txt" 256))

(def block-size 32)
(def n-elems     (count hashes-cpu))
(def matches-cpu (int-array (inc n-elems)))
(def results-cpu (int-array n-elems))
(def tmp-cpu     (int-array (/ n-elems 2 block-size)))

(cc.core/init)
(def gpu (cc.core/device 0))
(def ctx (atom (context gpu)))

(comment
  (commons.core/release @ctx)
  (reset! ctx nil))


(cc.core/in-context @ctx
  (def hashes-gpu  (cc.core/mem-alloc (* 8 n-elems)))
  (cc.core/memcpy-host! hashes-cpu hashes-gpu)
  
  (def matches-gpu (cc.core/mem-alloc (* 4 (count matches-cpu))))
  (def results-gpu (cc.core/mem-alloc (* 4 (count results-cpu))))
  (def tmp-gpu     (cc.core/mem-alloc (* 4 (count tmp-cpu)))))


(cc.core/in-context @ctx
  ; Ref. http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf for the cumulative sum code.
  ; Pull requests are welcome on how to utilize Thrust's copy_if ;)
  (def fn1
    (-> (str "
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
      
      extern \"C\" __global__ void f1(const int n, const uint64 *hashes, const uint64 target, const int max_dist, int *matches, int *results, int *tmp) {
        __shared__ int scan_array[2 * " block-size "];
        const int i_base = 2 * (blockIdx.x * blockDim.x + threadIdx.x), s_base = 2 * threadIdx.x;
        
        if (i_base + 0 < n) {
          const int matchIndicator = hammingDistance64(hashes[i_base + 0], target) <= max_dist;
          matches[i_base + 0]    = matchIndicator;
          scan_array[s_base + 0] = matchIndicator;
        }
        else {
          scan_array[s_base + 0] = 0;
        }
        
        if (i_base + 1 < n) {
          const int matchIndicator = hammingDistance64(hashes[i_base + 1], target) <= max_dist;
          matches[i_base + 1]    = matchIndicator;
          scan_array[s_base + 1] = matchIndicator;
        }
        else {
          scan_array[s_base + 1] = 0;
        }
        
        int stride;
        
        stride = 1;
        while (stride <= " block-size ") {
          const int index = (threadIdx.x + 1) * stride * 2 - 1;
          
          if(index < 2 * " block-size ") {
            scan_array[index] += scan_array[index - stride];
          }
          
          stride *= 2;
          __syncthreads();
        }
        
        stride = " block-size " / 2;
        while (stride > 0) {
          const int index = (threadIdx.x + 1) * stride * 2 - 1;
          
          if(index + stride < 2 * " block-size ") {
            scan_array[index + stride] += scan_array[index];
          }
          
          stride /= 2;
          __syncthreads();
        }
        
        if (i_base + 0 < n) {
          results[i_base + 0] = scan_array[s_base + 0];
        }
        
        if (i_base + 1 < n) {
          results[i_base + 1] = scan_array[s_base + 1];
        }
        
        if (threadIdx.x == 0) {
          tmp[blockIdx.x] = scan_array[2 * " block-size " - 1];
        }
      }
    ")
     cc.core/program cc.core/compile! cc.core/module (cc.core/function "f1")))

  (def fn2
    (-> "
      extern \"C\" __global__ void f2(const int n, int *results, int *tmp) {
        const int
          i = blockIdx.x * blockDim.x + threadIdx.x,
          base_value = tmp[blockIdx.x / 2];
        
        if (i < n) {
          results[i] += base_value;
        }
      }
    " cc.core/program cc.core/compile! cc.core/module (cc.core/function "f2"))))



(cc.core/in-context @ctx
  (let [target (nth hashes-cpu 3)
        max-dist 25]
    (cc.core/launch! fn1
      (cc.core/grid-1d (/ n-elems 2) block-size)
      (cc.core/parameters n-elems hashes-gpu target max-dist matches-gpu results-gpu tmp-gpu)))
  
  (cc.core/memcpy-host! matches-gpu matches-cpu)
; (cc.core/memcpy-host! results-gpu results-cpu)
  
  ; TODO: Do all the calculations in CUDA, we can only use at max 1024 threads per block!
  ; Or maybe loop more on GPU instead of full-fetched multi-kernel cumulative sum implementation.
  (cc.core/memcpy-host! tmp-gpu tmp-cpu)
  (cc.core/memcpy-host!
    (->> tmp-cpu (cons 0) (take (/ n-elems 2 block-size)) (reductions +) int-array)
    tmp-gpu)

  (cc.core/launch! fn2
    (cc.core/grid-1d n-elems block-size)
    (cc.core/parameters n-elems results-gpu tmp-gpu))
  
  (cc.core/memcpy-host! results-gpu results-cpu))


(comment
  (seq matches-cpu)
  (seq results-cpu)
  (seq tmp-cpu)

  (map * matches-cpu
    (reductions + matches-cpu)))





