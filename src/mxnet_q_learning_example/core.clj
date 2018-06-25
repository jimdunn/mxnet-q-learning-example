(ns mxnet-q-learning-example.core
  (:require [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.initializer :as initializer]))

;; Cliff walking Gridworld environment from Barto & Sutton's book.

(def cliffwalking-gridworld
  {:start 36
   :goal 47
   :actions {:up 0 :down 1 :right 2 :left 3}
   :top-edge #{0 1 2 3 4 5 6 7 8 9 10 11}
   :bottom-edge #{36 37 38 39 40 41 42 43 44 45 46 47}
   :right-edge #{11 23 35 47}
   :left-edge #{0 12 24 36}
   :cliff #{37 38 39 40 41 42 43 44 45 46}})

(defn grid-step [state action]
  (let [{:keys [top-edge bottom-edge left-edge right-edge
                actions start cliff goal]} cliffwalking-gridworld
        {:keys [up down left right]} actions
        next-state (cond
                     ;; attempted moves off grid don't change state
                     (or (and (= action up) (top-edge state))
                         (and (= action down) (bottom-edge state))
                         (and (= action right) (right-edge state))
                         (and (= action left) (left-edge state))) state
                     (= action up) (- state 12)
                     (= action down) (+ state 12)
                     (= action right) (inc state)
                     (= action left) (dec state))
        cliff? (some #(= next-state %) cliff)
        goal? (= next-state goal)]
    {:action action :state state
     :reward (if cliff? -100.0 -1.0)
     :next-state next-state
     :goal-state? goal?
     :terminal-state? (or cliff? goal?)}))

(def num-features 48)
(def num-actions 4)
(def eps 0.05)
(def gamma 0.99)

;; just a single layer network
(defn build-network []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden num-actions})
    (sym/linear-regression-output "lr" {:data data})))

(defn predict [mod features]
  (let [data (ndarray/array features [1 num-features])]
    (-> mod
        (m/predict {:eval-data (mx-io/ndarray-iter [data])})
        first
        ndarray/->vec)))

(defn train [mod features target]
  (let [data (ndarray/array features [1 num-features])
        label (ndarray/array target [1 num-actions])
        train-data (mx-io/ndarray-iter [data]
                                       {:label [label]
                                        :label-name "lr_label"
                                        :data-batch-size 1})]
    (-> mod
        (m/forward (mx-io/next train-data))
        (m/backward [label])
        (m/update))))

(defn eps-greedy-select [qs]
  (if (> (rand) eps)
    (first (apply max-key second (map-indexed vector qs)))
    (rand-int (count qs))))

(defn state->features [state]
  (assoc (into [] (repeat num-features 0.0)) state 1.0))

(defn target-vec [reward action terminal-state? qs next-qs]
  (let [target (if terminal-state?
                 reward
                 (+ reward (* gamma (apply max next-qs))))]
    (assoc qs action target)))

(defn cw-episode [mod]
  (loop [state (:start cliffwalking-gridworld)
         terminal? false
         goal? false
         step 0
         rewards 0.0]
    (if terminal?
      {:steps step :goal? goal?
       :rewards rewards}
      (let [features (state->features state)
            qs (predict mod features)
            action (eps-greedy-select qs)
            transition (grid-step state action)
            {:keys [reward next-state terminal-state? goal-state?]} transition
            next-features (state->features next-state)
            next-qs (predict mod next-features)
            target (target-vec reward action terminal-state? qs next-qs)]
        (train mod features target)
        (recur next-state
               terminal-state?
               goal-state?
               (inc step)
               (+ rewards reward))))))

(defn train-cw-network [num-episodes]
  (let [s (build-network)
        contexts [(context/default-context)]
        mod (-> (m/module s ["data"] ["lr_label"] contexts)
                (m/bind {:data-shapes [{:name "data" :shape [1 num-features]}]
                         :label-shapes [{:name "lr_label" :shape [1 num-actions]}]})
                (m/init-params {:initializer (initializer/xavier)})
                (m/init-optimizer {:optimizer (optimizer/sgd
                                               {:learning-rate 0.2})}))]
    (loop [episode 0
           episode-rewards []
           episode-steps []
           successes 0]
      (if (= episode num-episodes)
        {:successes successes
         :episode-rewards episode-rewards
         :episode-steps episode-steps}
        (let [{:keys [steps goal? rewards]} (cw-episode mod)]
          (recur (inc episode)
                 (conj episode-rewards rewards)
                 (conj episode-steps steps)
                 (if goal? (inc successes) successes)))))))
