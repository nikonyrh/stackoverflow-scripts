(defproject tweetable-hash "0.0.1-SNAPSHOT"
  :description "FIXME: write description"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [com.climate/claypoole "1.1.4"]]
  :javac-options ["-target" "1.6" "-source" "1.6" "-Xlint:-options"]
  :aot [tweetable-hash.core]
  :main tweetable-hash.core)
  