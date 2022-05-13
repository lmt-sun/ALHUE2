
from math import sqrt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np


# 获取数据
from canonical_ensemble import UnderBagging, SMOTEBagging

data_l = pd.read_csv("netbsd52.csv", header=None,
                     names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                            '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43',
                            '44', '45', '46', '47', '48', '49', '50', '51', '52'])
data = data_l.iloc[:, :52]
target = data_l.iloc[:, 52]
data = DataFrame(data)
target = DataFrame(target)

class ensembles:
    def __init__(self, method):
        self.method = method

    def fit(self, data, target):
        # 训练集、测试集的划分
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3)
        #train, test, labled, unlab = split(X=data, y=target, test_ratio=0.3, initial_label_rate=0.1,
         #                                  split_count=1, all_class=True, saving_path='..')
        #unlab = unlab[0].tolist()
        #labled = labled[0].tolist()
        #test = test[0].tolist()

        # 基于信息量和代表性的查询策略
        #for i in range(71):
         # Q = QueryInstanceQUIRE(X=data, y=target, train_idx=train[0])
          #select = Q.select(labled, unlab)
          #unlab.remove(select[0])
          #labled.append(select[0])

        #train_data = data.loc[labled]
        #train_target = target.loc[labled]
        #train_data = DataFrame(train_data)
        #train_target = DataFrame(train_target)
        #test_data = data.loc[test]
        #test_target = target.loc[test]
        #test_data = DataFrame(test_data)
        #test_target = DataFrame(test_target)

        return train_data, train_target, test_data, test_target

    def predict(self, data, target):
      Bal_sum = []
      fm_sum = []
      auc_sum = []
      for i in range(10):
        train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3)

        if self.method == 'SMOTEBagging':
           smotebagging = SMOTEBagging().fit(train_data, train_target)
           y = smotebagging.predict(test_data)
           y = pd.DataFrame(y)
           predict_y = y.astype(int)
           #评估
           racall1 = recall_score(test_target, predict_y, average='macro')
           one, two = confusion_matrix(y_true=test_target, y_pred=predict_y)
           TP = one[0]
           FN = one[1]
           FP = two[0]
           TN = two[1]
           PF = FP / (FP + TN)
           bal = 1 - (sqrt(pow(PF, 2) + pow((1 - racall1), 2)) / sqrt(2))
           PD = TP / (TP + FN)
           f1 = f1_score(test_target, predict_y, average='macro')
           auc = roc_auc_score(test_target, predict_y)
           Bal_sum.append(bal)
           fm_sum.append(f1)
           auc_sum.append(auc)
           print(bal)
           print(f1)
           print(auc)
           print()

        if self.method == 'RUSBagging':
            rusbagging = UnderBagging().fit(train_data, train_target)
            y = rusbagging.predict(test_data)
            y = pd.DataFrame(y)
            predict_y = y.astype(int)
            #评估
            racall1 = recall_score(test_target, predict_y, average='macro')
            one, two = confusion_matrix(y_true=test_target, y_pred=predict_y)
            TP = one[0]
            FN = one[1]
            FP = two[0]
            TN = two[1]
            PF = FP / (FP + TN)
            bal = 1 - (sqrt(pow(PF, 2) + pow((1 - racall1), 2)) / sqrt(2))
            PD = TP / (TP + FN)
            f1 = f1_score(test_target, predict_y, average='macro')
            auc = roc_auc_score(test_target, predict_y)
            Bal_sum.append(bal)
            fm_sum.append(f1)
            auc_sum.append(auc)
            print(bal)
            print(f1)
            print(auc)
            print()

        if self.method == 'SMOTE':
            smo = SMOTE(random_state=42, k_neighbors=1)
            train_data, train_target = smo.fit_sample( train_data, train_target)
            #estimator = DecisionTreeClassifier(criterion="gini", max_depth=12, class_weight="balanced")
            #estimator = KNeighborsClassifier(n_neighbors=3)
            estimator = LinearSVC(random_state=74)
            #estimator = LogisticRegression()
            #estimator = RandomForestClassifier(n_estimators=50)
            #estimator = GaussianNB()
            estimator.fit(train_data, train_target)
            # 模型评估
            predict_y = estimator.predict(test_data)
            # 评估
            racall = recall_score(test_target, predict_y, average='macro')
            one, two = confusion_matrix(y_true=test_target, y_pred=predict_y)
            TP = one[0]
            FN = one[1]
            FP = two[0]
            TN = two[1]
            PF = FP / (FP + TN)
            bal = 1 - (sqrt(pow(PF, 2) + pow((1 - racall), 2)) / sqrt(2))
            PD = TP / (TP + FN)
            f1 = f1_score(test_target, predict_y, average='macro')
            auc = roc_auc_score(test_target, predict_y)
            Bal_sum.append(bal)
            fm_sum.append(f1)
            Bal_sum.append(bal)
            fm_sum.append(f1)
            auc_sum.append(auc)
            print(bal)
            print(f1)
            print(auc)
            print()

        if self.method == 'RUS':
            rus = RandomUnderSampler(random_state=0)
            train_data, train_target = rus.fit_resample(train_data, train_target)
            #estimator = DecisionTreeClassifier(criterion="gini", max_depth=12, class_weight="balanced")
            #estimator = KNeighborsClassifier(n_neighbors=3)
            estimator = LinearSVC(random_state=74)
            #estimator = LogisticRegression()
            #estimator = RandomForestClassifier(n_estimators=50)
            #estimator = GaussianNB()
            estimator.fit(train_data, train_target)
            # 模型评估
            predict_y = estimator.predict(test_data)
            # 评估
            racall = recall_score(test_target, predict_y, average='macro')
            one, two = confusion_matrix(y_true=test_target, y_pred=predict_y)
            TP = one[0]
            FN = one[1]
            FP = two[0]
            TN = two[1]
            PF = FP / (FP + TN)
            bal = 1 - (sqrt(pow(PF, 2) + pow((1 - racall), 2)) / sqrt(2))
            PD = TP / (TP + FN)
            f1 = f1_score(test_target, predict_y, average='macro')
            auc = roc_auc_score(test_target, predict_y)
            Bal_sum.append(bal)
            fm_sum.append(f1)
            auc = roc_auc_score(test_target, predict_y)
            Bal_sum.append(bal)
            fm_sum.append(f1)
            auc_sum.append(auc)
            print(bal)
            print(f1)
            print(auc)
            print()

      print("bal:")
      print(np.mean(Bal_sum))
      print("f1:")
      print(np.mean(fm_sum))
      print("auc:")
      print(np.mean(auc_sum))



if __name__ == '__main__':
    data = data
    target = target
    type = 'SMOTEBagging'
    ensembles = ensembles(type)
    ensembles.predict(data, target)