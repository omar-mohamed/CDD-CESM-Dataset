import json
import tensorflow.keras.backend as kb
import numpy as np
import os
import shutil
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score, hamming_loss, \
    roc_auc_score


class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """

    def __init__(self, sequence, class_names, weights_path, output_weights_path, confidence_thresh=0.5, stats=None,
                 workers=1):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.weights_path = weights_path
        self.confidence_thresh = confidence_thresh
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"{os.path.split(output_weights_path)[1]}",
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print(f"current learning rate: {self.stats['lr']}")

        """
        y_hat shape: (#samples, len(class_names))
        y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
        """
        y_hat = self.model.predict(self.sequence, workers=self.workers)
        y = self.sequence.get_y_true()

        print(f"*** epoch#{epoch + 1} dev auroc ***")
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print(f"{i + 1}. {self.class_names[i]}: {score}")
        print("*********************************")

        prec, rec, fscore, support = precision_recall_fscore_support(y, y_hat >= self.confidence_thresh,
                                                                     average='macro')
        AP = average_precision_score(y, y_hat)
        exact_accuracy = accuracy_score(y, y_hat >= self.confidence_thresh)
        ham_loss = hamming_loss(y, y_hat >= self.confidence_thresh)
        print(
            f"precision:{prec:.2f}, recall: {rec:.2f}, fscore: {fscore:.2f}, AP: {AP:.2f}, exact match accuracy: {exact_accuracy:.2f}, hamming loss: {ham_loss:.2f}")
        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print(f"mean auroc: {mean_auroc}")
        if mean_auroc > self.stats["best_mean_auroc"]:
            print(f"update best auroc from {self.stats['best_mean_auroc']} to {mean_auroc}")

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print(f"update log file: {self.best_auroc_log_path}")
            with open(self.best_auroc_log_path, "a") as f:
                f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc}, lr: {self.stats['lr']}\n")

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print(f"update model file: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_auroc"] = mean_auroc
            self.stats["AP"] = AP
            self.stats["precision"] = prec
            self.stats["recall"] = rec
            self.stats["fscore"] = fscore
            self.stats["hamming_loss"] = ham_loss
            self.stats["f1_score"] = ham_loss
            self.stats["exact_accuracy"] = exact_accuracy

            print("*********************************")
        else:
            print(f"best auroc is still {self.stats['best_mean_auroc']}")

        return
