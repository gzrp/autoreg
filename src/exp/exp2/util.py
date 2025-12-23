import json
import os
import time

import numpy as np
from ray.tune import Callback



def to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

class BufferedBestSampler(Callback):
    def __init__(self, exp_name="default", dataset="default", metric = "bacc", mode="max", max_epochs=4, start_time = None, log_file = "best_over_time.jsonl", flush_every=100, verbose=True):
        super(BufferedBestSampler, self).__init__()
        if start_time is None:
            raise ValueError("You MUST provide start_time from outside!")
        self.start_time = start_time
        self.metric = metric
        self.mode = mode
        self.exp_name = exp_name
        self.dataset = dataset

        self.max_epochs = max_epochs
        self.best_metric = None  # 全局最佳 metric
        self.best_config = None  # 全局最佳 config
        self.auc_history = None
        self.loss_history = None
        self.verbose = verbose
        self.default_log_dir = f"/data/ruipeng/workdir/autoreg/.exp_results/{self.metric}/logs/{self.dataset}/{self.exp_name}/"
        os.makedirs(self.default_log_dir, exist_ok=True)
        self.log_file = os.path.join(self.default_log_dir, log_file)
        self.flush_every = flush_every
        self.buffer = []

    def _is_better(self, v):
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return v > self.best_metric
        return v < self.best_metric

    def _flush_buffer(self):
        if not self.buffer:
            return
        with open(self.log_file, "a") as f:
            for row in self.buffer:
                f.write(json.dumps(row, default=to_serializable) + "\n")
        self.buffer = []
        if self.verbose:
            print(f"[Flush] {self.log_file} written.")


    def on_trial_result(self, iteration, trials, trial, result, **info):
        # 只接受完整的 trial
        training_iter = result.get("training_iteration")
        if training_iter < self.max_epochs:
            return

        now = time.time()
        elapsed = now - self.start_time
        metric_value = result.get(self.metric)
        if metric_value is None:
            return

        # 与全局最佳进行比较
        if self._is_better(metric_value):
            self.best_metric = metric_value
            self.best_config = trial.config
            self.auc_history = result.get("auc_history")
            self.loss_history = result.get("loss_history")

        # 写入到缓冲区而不是文件
        record = {
            "epoch": training_iter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "elapsed_time": elapsed,
            "best_metric": self.best_metric,
            "auc_history": self.auc_history,
            "loss_history": self.loss_history,
            "best_config": self.best_config,
        }

        self.buffer.append(record)
        if self.verbose:
            print(f"[Result]{record}")

        # 根据条目数量触发flush
        if len(self.buffer) >= self.flush_every:
            self._flush_buffer()

    def on_experiment_end(self, **info):
        # 确保全部写入
        self._flush_buffer()
        if self.verbose:
            print("[End] Final flush completed.")



class BufferedBestSamplerBacc(Callback):
    def __init__(self, exp_name="default", dataset="default", metric = "bacc", mode="max", max_epochs=4, start_time = None, log_file = "best_over_time.jsonl", flush_every=100, verbose=True):
        super(BufferedBestSamplerBacc, self).__init__()
        if start_time is None:
            raise ValueError("You MUST provide start_time from outside!")
        self.start_time = start_time
        self.metric = metric
        self.mode = mode
        self.exp_name = exp_name
        self.dataset = dataset

        self.max_epochs = max_epochs
        self.best_metric = None  # 全局最佳 metric
        self.best_config = None  # 全局最佳 config
        self.bacc_history = None
        self.loss_history = None
        self.verbose = verbose
        self.default_log_dir = f"/data/ruipeng/workdir/autoreg/.exp_results/{self.metric}/logs/{self.dataset}/{self.exp_name}/"
        os.makedirs(self.default_log_dir, exist_ok=True)
        self.log_file = os.path.join(self.default_log_dir, log_file)
        self.flush_every = flush_every
        self.buffer = []

    def _is_better(self, v):
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return v > self.best_metric
        return v < self.best_metric

    def _flush_buffer(self):
        if not self.buffer:
            return
        with open(self.log_file, "a") as f:
            for row in self.buffer:
                f.write(json.dumps(row, default=to_serializable) + "\n")
        self.buffer = []
        if self.verbose:
            print(f"[Flush] {self.log_file} written.")


    def on_trial_result(self, iteration, trials, trial, result, **info):
        # 只接受完整的 trial
        training_iter = result.get("training_iteration")
        if training_iter < self.max_epochs:
            return

        now = time.time()
        elapsed = now - self.start_time
        metric_value = result.get(self.metric)
        if metric_value is None:
            return

        # 与全局最佳进行比较
        if self._is_better(metric_value):
            self.best_metric = metric_value
            self.best_config = trial.config
            self.bacc_history = result.get("bacc_history")
            self.loss_history = result.get("loss_history")

        # 写入到缓冲区而不是文件
        record = {
            "epoch": training_iter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "elapsed_time": elapsed,
            "best_metric": self.best_metric,
            "bacc_history": self.bacc_history,
            "loss_history": self.loss_history,
            "best_config": self.best_config,
        }

        self.buffer.append(record)
        if self.verbose:
            print(f"[Result]{record}")

        # 根据条目数量触发flush
        if len(self.buffer) >= self.flush_every:
            self._flush_buffer()

    def on_experiment_end(self, **info):
        # 确保全部写入
        self._flush_buffer()
        if self.verbose:
            print("[End] Final flush completed.")
