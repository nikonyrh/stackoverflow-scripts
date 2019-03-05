(ns tweetable-hash.core
  (:require [com.climate.claypoole :as cp])
  (:gen-class))

(set! *warn-on-reflection* true)

(def small-primes
  (->> (range 101 1000 2)
       (filter #(.isProbablePrime (BigInteger/valueOf %) 7))
       vec))

(def big-primes
  (->> (range (- 16rffffff 100) 16777213 2)
       (filter #(.isProbablePrime (BigInteger/valueOf %) 7))
       (concat [16rffffff])
       vec))

(comment
  ; Best found parameters for these few distinct function constructs.
  (def f #(reduce(fn[r i](mod(+(* r 23)i)16777213))(map *(cycle(map int"ZtabAR%H|-KrykQn{]u9f:F}v#OI^so3$x54z2&gwX<S~"))(for[c %](bit-xor(int c)3))))) ; 6148
  (def f #(reduce(fn[r i](mod(+(* r 263)i)16777213))(map *(cycle(map int"i@%(J|IXt3&R5K'XOoa+Qk})w<!w[|3MJyZ!=HGzowQlN"))(map int %)(rest(range)))))   ; 6021
  (def f #(reduce(fn[r i](mod(+(* r 811)i)16777213))(map *(cycle(map int"~:XrBaXYOt3'tH-x^W?-5r:c+l*#*-dtR7WYxr(CZ,R6J7=~vk"))(map int %))))           ; 6019
  
  (let [rows (->> "/home/wrecked/projects/stackoverflow-scripts/76781_tweetable_hash/tweetable-hash/english.txt" slurp (re-seq #"[^\r\n]+"))]
    (for [limit [8 12 16 50]]
      [limit
       (->> rows
            (filter #(<= (count %) limit))
            (map f)
            frequencies
            vals
            (remove #{1})
            (reduce +))])))


(defn search [fname n-proc word-len seed]
  ; The idea is to first find optimal prefix for the seed so that words shorter than N
  ; characters have the least number of collisions. Later we'll advance  to process
  ; longer and longer strings, hoping to find a good seed value.
  (let [f         #(reduce(fn[r i](mod(+(* r %)i)%2))(map *(cycle(map int %3))(map int %4)))
        best-score (atom 1e6)
        rows    (->> fname slurp (re-seq #"[^\r\n]+") (filter #(<= (count %) word-len))
                     (sort-by count) vec)
        gen-len (- word-len (count seed))
        params  (for [c [(->> (range 32 127) (map char) (remove (set "\\`\"\n")) vec)]
                      rand-char [#(rand-nth c)
                                 i (range)]]
                  [(rand-nth small-primes)
                   (rand-nth big-primes)
                   (->> rand-char (repeatedly gen-len) (apply str seed))])]
    (->> (cp/pfor n-proc [p params]
           ; Loop until we have hashed every row, or the total running score is
           ; worse than the current best score.
           (let [f (apply partial f p)
                 max-score @best-score]
             (loop [score 0 seen-hashes {} rows rows]
               (if-let [row (first rows)]
                 (let [hash        (f row)
                       seen-count  (int (get seen-hashes hash 0))
                       score       (+ score (case seen-count 0 0 1 2 (- seen-count 1)))
                       seen-hashes (assoc seen-hashes hash (inc seen-count))]
                   (if (< score max-score)
                     (recur score seen-hashes (rest rows))))
                 [score p]))))
         ; Receive nils or [score p] tuples, keep track of the best score so far and print stats.
         (reduce (fn [i [score param]]
                   (do
                     (if (and score (< score @best-score))
                       (do
                         (println i (java.util.Date.) score param)
                         (swap! best-score #(min score %)))
                       (if (-> i (mod 1000) zero?)
                         (println i (java.util.Date.))))
                     (inc i)))
            0)
         time)))

; Example run:
;  java -jar tweetable-hash-0.0.1-SNAPSHOT-standalone.jar english.txt 70 16 '~:XrBaXY'
(defn -main [fname n-proc word-len seed]
  (do
    (search fname
      (Integer/parseInt n-proc)
      (Integer/parseInt word-len)
      seed)
    (shutdown-agents)
    (System/exit 0)))
