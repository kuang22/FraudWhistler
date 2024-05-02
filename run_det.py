import numpy as np
from sklearn import svm
import sys
import json
# Global data and code
with open("detection.json") as f:
  detection_config = json.load(f)
  dets = detection_config['dets']
  det_name_map = detection_config['det_name_map']
  aes = detection_config['aes']
  res_dir = detection_config['res_dir']


if __name__ == "__main__":
  fpr = int(sys.argv[1]) / 100
  sys.stdout = open(f"{res_dir}/det.txt", "w")
  X_train = {}
  X_test = {}
  with open(f"{res_dir}/scores_data.npy", "rb") as f:
    for det in dets:
      X_train[det] = np.load(f)
      X_test[det] = np.load(f)
    test_asr_succ = np.load(f)
  X_adv = {}
  X_succ_adv = {}
  with open(f"{res_dir}/scores_adv_data.npy", "rb") as f:
    for det in dets:
      X_adv[det] = {}
      X_succ_adv[det] = {}
      for alg in aes:
        X_adv[det][alg] = np.load(f)
        X_succ_adv[det][alg] = np.load(f)
  print("Accuracy(ori) on benign examples (ACC_be with no defense):",
        f"{np.count_nonzero(test_asr_succ) / test_asr_succ.size * 100:.2f}%")

  for det in dets:
    clf = svm.OneClassSVM(nu=fpr, kernel='rbf')
    clf.fit(X_train[det])
    y_pred_train = clf.predict(X_train[det])
    y_pred_test = clf.predict(X_test[det])
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_test_asr_succ = \
        y_pred_test[(y_pred_test == 1) * (test_asr_succ == 1)].size
    for alg in aes:
      exec(f"y_pred_{alg} = clf.predict(X_adv['{det}']['{alg}'])")
      exec(f"succ_len = X_succ_adv['{det}']['{alg}'].shape[0]")
      if succ_len == 0:
        exec(f"y_succ_pred_{alg} = np.array([])")
      else:
        exec(f"y_succ_pred_{alg} = \
                        clf.predict(X_succ_adv['{det}']['{alg}'])")
      exec(f"n_error_{alg} = y_pred_{alg}[y_pred_{alg} == 1].size")
      exec(f"n_succ_error_{alg} = \
                    y_succ_pred_{alg}[y_succ_pred_{alg} == 1].size")
    print("--------------")
    print(f"{det_name_map[det]}:")
    # print("Train Error:",
    #       f"{n_error_train / y_pred_train.size * 100: .2f}%")
    print("Accuracy on benign examples (ACC_be):",
          f"{n_test_asr_succ / y_pred_test.size * 100:.2f}%")

    print("Detector's accuracy on adversarial examples (ACC_ae):")
    total_acc = 0
    num_acc = 0
    for alg in aes:
      exec(f"acc = (1 - n_error_{alg} / y_pred_{alg}.size) * 100")
      total_acc += acc
      num_acc += 1
      # print(f"{alg.upper()}: {acc:.2f}%")
    print(f"Average: {total_acc / num_acc:.2f}%")

    print("Robust accuracy of whole system on AEs (ACC_rob):")
    total_acc = 0
    num_acc = 0
    for alg in aes:
      exec(f"acc = (1 - n_succ_error_{alg} / y_pred_{alg}.size) * 100")
      total_acc += acc
      num_acc += 1
      # print(f"{alg.upper()}: {acc:.2f}%")
    print(f"Average: {total_acc / num_acc:.2f}%")
