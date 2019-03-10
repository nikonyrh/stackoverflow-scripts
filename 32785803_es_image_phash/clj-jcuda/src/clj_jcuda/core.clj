(ns clj-jcuda.core
  (:require [uncomplicate.commons.core :as commons.core]
            [uncomplicate.clojurecuda.core :as cc.core :refer [with-context context]]            
            [clojure.repl :refer [source doc]]
            [clojure.pprint :refer [pprint]]
            
            [nikonyrh-utilities-clj.core :as u]))

(set! *warn-on-reflection* true)


(defn read-numbers [fname]
  (let [; Well this seems a bit stupid way of converting a 64-bit unsigned int to a signed one :o
        max-long (apply * (repeat 63 2N))
        to-long #(long (if (>= % max-long) (- max-long %) %))]
    (with-open [rdr (clojure.java.io/reader fname)]
      (->> rdr line-seq (map (fn [^java.lang.String s] (-> s java.math.BigInteger. to-long)))
           long-array))))

; This lazy seq is cached, which makes it faster to extend the test dataset size.
(def all-hashes
 (->> "/home/wrecked/projects/stackoverflow-scripts/32785803_es_image_phash/py/"
       clojure.java.io/file
       file-seq
       (map str)
       (filter (fn [^java.lang.String f] (.endsWith f ".txt")))
       sort
       (map read-numbers)))

(cc.core/init)
(def gpu (cc.core/device 0))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def n-elems (int 80e6))

; Refer to py/generate_to_file.py on how to generate these files.
(def hashes-cpu
  (->> all-hashes
       (apply concat)
       (take n-elems)
       long-array))

(assert (= n-elems (count hashes-cpu)))

(def block-size 1024)
(def matches-cpu (int-array (inc n-elems)))
(def results-cpu (int-array n-elems))
(def n-tmp       (int (Math/ceil (/ n-elems 2 block-size))))
(def tmp-cpu     (int-array n-tmp))


(def ctx (atom (context gpu)))

(comment
  (commons.core/release @ctx)
  (reset! ctx nil)
  "Released the ctx")

(time
  (cc.core/in-context @ctx
    (def hashes-gpu  (cc.core/mem-alloc (* 8 n-elems)))
    (cc.core/memcpy-host! hashes-cpu hashes-gpu)
    
    (def matches-gpu (cc.core/mem-alloc (* 4 (count matches-cpu))))
    (def results-gpu (cc.core/mem-alloc (* 4 (count results-cpu))))
    (def tmp-gpu     (cc.core/mem-alloc (* 4 (count tmp-cpu))))))


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
    (-> (str "
      extern \"C\" __global__ void f2(const int n, const int *hashes, int *matches, const int *cumsum, const int *block_cumsum) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i < n && matches[i]) {
          const int match_ix = cumsum[i] + block_cumsum[blockIdx.x / 2];
          matches[match_ix] = hashes[i];
        }
        
        if (i >= n - 1 && threadIdx.x == " block-size " - 1) {
          // Storing the number of matches
          matches[0] = cumsum[i] + block_cumsum[blockIdx.x / 2];
        }
      }
    ")
     cc.core/program cc.core/compile! cc.core/module (cc.core/function "f2"))))


(defn cumsum! [^ints iarr]
  (let [n (alength iarr)]
    (loop [sum 0 i 0]
      (when (< i n)
        (let [sum (+ sum (aget iarr i))]
          (aset iarr i sum)
          (recur sum (inc i)))))))

  


(defn find-matches [target max-dist]
  (cc.core/in-context @ctx
    (let [times []
          
          times (conj times (System/nanoTime)) ; step1
          _ (cc.core/launch! fn1
              (cc.core/grid-1d (/ n-elems 2) block-size)
              (cc.core/parameters n-elems hashes-gpu target max-dist matches-gpu results-gpu tmp-gpu))
          
          ; _ (do (cc.core/memcpy-host! matches-gpu matches-cpu) (cc.core/memcpy-host! results-gpu results-cpu))]
          
          ; TODO: Do all the calculations in CUDA, we can only use at max 1024 threads per block!
          ; Or maybe loop more on GPU instead of full-fetched multi-kernel cumulative sum implementation.
          times (do (cc.core/synchronize!) (conj times (System/nanoTime))) ; step2
          _ (cc.core/memcpy-host! tmp-gpu tmp-cpu)
          
          times (do (cc.core/synchronize!) (conj times (System/nanoTime))) ; step3
        ; tmp-cumsum-cpu (->> tmp-cpu (cons 0) (take n-tmp) (reductions +) int-array)
          tmp-cumsum-cpu (->> tmp-cpu (cons 0) (take n-tmp) int-array)
          _ (cumsum! tmp-cumsum-cpu)
          
          times (conj times (System/nanoTime)) ; step4
          _ (cc.core/memcpy-host! tmp-cumsum-cpu tmp-gpu)
          
          times (do (cc.core/synchronize!) (conj times (System/nanoTime))) ; step5
          _ (cc.core/launch! fn2
              (cc.core/grid-1d n-elems block-size)
              (cc.core/parameters n-elems hashes-gpu matches-gpu results-gpu tmp-gpu))
          
          times (do (cc.core/synchronize!) (conj times (System/nanoTime))) ; step6
          _ (cc.core/memcpy-host! matches-gpu matches-cpu 4)                                 ; The number of matches
          _ (cc.core/memcpy-host! matches-gpu matches-cpu (-> matches-cpu first inc (* 4))) ; The number of matches & hashes themselves
          times (do (cc.core/synchronize!) (conj times (System/nanoTime)))  ; step7
          
          _ nil]
      [times matches-cpu])))


(comment
  (def n-tests 1000)
  
  (time
    (def results
      (let [max-dist 8] 
        (->> (for [target (->> (repeatedly #(rand-nth hashes-cpu)) (take n-tests))]
               (let [[times matches] (find-matches target max-dist)]
                 (merge
                   {:n-matches (first matches)}
                   (let [time-deltas (->> (map - (rest times) times) (map #(* 1e-6 %)))]
                     (->> time-deltas
                          (cons (apply + time-deltas))
                          (zipmap (map (comp keyword str) (repeat "step") (range))))))))
             vec))))
  
  (first results)
  
  
  (let [fields (-> results first keys sort)
        ps     (for [p [25 50 75 95]] (-> p (* (dec n-tests) 0.01) Math/round int))]
    (->> results
         (map (apply juxt fields))
         (apply map list)
         (map #(->> ps
                   (map (-> % sort vec))
                   (map (fn [v] (-> v (* 1000.0) Math/round bigdec (* 0.001M) double)))))
         (zipmap fields)
         (merge
           (sorted-map)
           {:n-elems (* n-elems 1e-6)})
         pprint)))


(comment
  (->> results
       (map #(-> % :n-matches (Math/pow 0.25) Math/round (Math/pow 4.0) int))
       frequencies (into (sorted-map)) pprint)
  
  (into
    {:n-elems (* n-elems 1e-6)}
    (->> results
         (apply merge-with +)
         (map (fn [[k v]] [k (-> v (/ n-tests) double)])))))


(comment
  (seq matches-cpu)
  (seq results-cpu)
  (seq hashes-cpu)
  (seq tmp-cpu)
  
  (take (first matches-cpu) (rest matches-cpu)))


(comment
  (def benchmarks
    [[:v1 ; Multiple calls to synchronize! adds about 1 ms to the total time.
      '[{:n-elems 10.0,
         :n-matches (13569.0 22063.0 35488.0 67179.0),
         :step0 (2.567 2.659 2.832 3.87),
         :step1 (1.592 1.603 1.73 2.393),
         :step2 (0.014 0.017 0.022 0.056),
         :step3 (0.466 0.473 0.495 0.727),
         :step4 (0.012 0.014 0.046 0.635),
         :step5 (0.407 0.437 0.476 0.577),
         :step6 (0.028 0.037 0.051 0.086)}
        
        {:n-elems 20.0,
         :n-matches (28101.0 44797.0 67210.0 124425.0),
         :step0 (5.886 6.259 6.856 7.526),
         :step1 (3.756 3.932 4.546 5.223),
         :step2 (0.022 0.025 0.047 0.075),
         :step3 (0.939 0.953 1.099 1.47),
         :step4 (0.019 0.032 0.066 0.418),
         :step5 (0.912 0.943 0.963 1.126),
         :step6 (0.042 0.056 0.078 0.136)}
        
        {:n-elems 40.0,
         :n-matches (54070.0 87549.0 136194.0 249672.0),
         :step0 (11.37 11.887 12.296 12.885),
         :step1 (7.13 7.716 8.298 8.54),
         :step2 (0.034 0.053 0.057 0.087),
         :step3 (1.882 1.911 2.134 2.697),
         :step4 (0.028 0.032 0.045 0.082),
         :step5 (1.72 1.864 1.919 2.607),
         :step6 (0.065 0.091 0.132 0.222)}
        
        {:n-elems 80.0,
         :n-matches (114772.0 188799.0 290454.0 574165.0),
         :step0 (24.442 24.972 25.528 26.186),
         :step1 (16.472 16.584 16.682 16.937),
         :step2 (0.069 0.071 0.075 0.105),
         :step3 (3.778 3.828 4.259 5.279),
         :step4 (0.046 0.052 0.066 0.117),
         :step5 (3.721 3.873 4.278 5.027),
         :step6 (0.123 0.172 0.242 0.406)}]]])
  
  
  (->> benchmarks
       first second
       
       ; Calculating the scan speed in terms of millions of hashes / second
       (map #(/ (-> % :n-elems)
                (-> % :step0 (nth 2) (* 1e-3))))))

