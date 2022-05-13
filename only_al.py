
from math import sqrt
from alipy.data_manipulate import split
from alipy.query_strategy.query_labels import QueryInstanceUncertainty, _get_proba_pred, QueryInstanceQBC, QueryInstanceQUIRE, QureyExpectedErrorReduction
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from pandas.core.frame import DataFrame
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV



# 获取数据
data_l = pd.read_csv("mysql52.csv", header=None,
                     names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                            '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43',
                            '44', '45', '46', '47', '48', '49', '50', '51', '52'])
data = data_l.iloc[:, :52]
target = data_l.iloc[:, 52]
data = DataFrame(data)
target = DataFrame(target)


class SKA:
    def __init__(self, n_components_num):
        self.n_components_num = n_components_num

    def fit(self, data, target):

        train, test, labled, unlab = split(X=data, y=target, test_ratio=0.3, initial_label_rate=0.1,
                                           split_count=1, all_class=True, saving_path='..')
        unlab = unlab[0].tolist()
        labled = labled[0].tolist()

        # 随机选取未标记的数据
        #for i in range(1530):
         #  ran = random.randint(0, len(unlab) - 1)
          # labled.append(unlab[ran])
           #unlab.remove(unlab[ran])

        # 基于代表性和信息性的的的查询策略
        for i in range(212):
          Q = QueryInstanceQUIRE(X=data, y=target, train_idx=train[0])
          select = Q.select(labled, unlab)
          labled.append(select[0])
          unlab.remove(select[0])
          print(select[0])

        #基于不确定性
        #Q = QueryInstanceUncertainty(X=data, y=target, measure='margin')
        #select = Q.select(labled, unlab, batch_size=1530)
        #for i in range(1530):
         # labled.append(select[i])
        #sampler = hierarchical_clustering_AL.HierarchicalClusterAL(X_train, y_train, seed=SEED)
        #print(labled)

        train_data = data.loc[labled]
        train_target = target.loc[labled]
        train_data = DataFrame(train_data)
        train_target = DataFrame(train_target)
        return data, train_data, train_target, test[0]

    def predict(self, predict_type, data, target):
      #bal_sum = []
      #fm_sum = []
      #auc_sum = []
     for i in range(20):
        datanew, smo_train_data, smo_train_target, test_index = self.fit(data, target)
        test_data = datanew.loc[test_index]
        test_target = target.loc[test_index]
        if predict_type == 'DT':
          # 1、使用决策树预估器分类
          estimator1 = DecisionTreeClassifier(criterion="gini", max_depth=12, class_weight="balanced")
          estimator1.fit(smo_train_data, smo_train_target)

        # 模型评估
          y_predict1 = estimator1.predict(test_data)
          racall1 = recall_score(test_target, y_predict1, average='macro')
        # b、计算准确率
          one, two = confusion_matrix(y_true=test_target, y_pred=y_predict1)
          TP = one[0]
          FN = one[1]
          FP = two[0]
          TN = two[1]
          PF = FP / (FP + TN)
          bal1 = 1 - (sqrt(pow(PF, 2) + pow((1 - racall1), 2)) / sqrt(2))
          auc = roc_auc_score(test_target, y_predict1)
          fm = f1_score(test_target, y_predict1, average='macro')
          print("DT_auc:\n", auc)
          print("DT_f-measure:\n", fm)
          print("DT_bal:\n", bal1)
        #bal_sum.append(bal1)
        #fm_sum.append(fm)
        #auc_sum.append(auc)
        #print("\n")


        if predict_type == 'KNN':
            # 2、KNN
          estimator2 = KNeighborsClassifier(n_neighbors=5)
          estimator2.fit(smo_train_data, smo_train_target)
            # 5、模型评估
            # 方法一、直接比对真实值和预测值
          y_predict2 = estimator2.predict(test_data)
          racall2 = recall_score(test_target, y_predict2, average='macro')
          one, two = confusion_matrix(y_true=test_target, y_pred=y_predict2)
          TP = one[0]
          FN = one[1]
          FP = two[0]
          TN = two[1]
          PF = FP / (FP + TN)
          bal2 = 1 - (sqrt(pow(PF, 2) + pow((1 - racall2), 2)) / sqrt(2))
          auc = roc_auc_score(test_target, y_predict2)
          fm = f1_score(test_target, y_predict2, average='macro')
          print("KNN_auc:\n", auc)
          print("KNN_f-measure2:\n", fm)
          print("KNN_bal:\n", bal2)
        #bal_sum.append(bal2)
        #fm_sum.append(fm)
        #auc_sum.append(auc)
        #print("\n")


        if predict_type == 'SVM':
            # 3 、SVM
          estimator3 = LinearSVC(random_state=88)
          estimator3.fit(smo_train_data, smo_train_target)
          y_predict3 = estimator3.predict(test_data)
          racall3 = recall_score(test_target, y_predict3, average='macro')
          one, two = confusion_matrix(y_true=test_target, y_pred=y_predict3)
          recall = recall_score(test_target, y_predict3, average='macro')
          one, two = confusion_matrix(y_true=test_target, y_pred=y_predict3)
          TP = one[0]
          FN = one[1]
          FP = two[0]
          TN = two[1]
          PF = FP / (FP + TN)
          PD = TP / (TP + FN)
          bal = 1 - (sqrt(pow(PF, 2) + pow((1 - recall), 2)) / sqrt(2))

          fm = f1_score(test_target, y_predict3, average='macro')
          # Accuracy evaluation
          accuracy = accuracy_score(test_target, y_predict3)
          AUC = roc_auc_score(test_target, y_predict3)
          print("SVM:\n")
          print("PF:\n",PF)
          print("PD:\n",PD)
          print("bal:\n", bal)
          print("AUC:\n",AUC)


        if predict_type == 'LR':
            # 4、LR
          estimator5 = LogisticRegressionCV(multi_class='ovr')
          estimator5.fit(smo_train_data, smo_train_target)
          y_predict5 = estimator5.predict(test_data)
          racall5 = recall_score(test_target, y_predict5, average='macro')
          one, two = confusion_matrix(y_true=test_target, y_pred=y_predict5)
          TP = one[0]
          FN = one[1]
          FP = two[0]
          TN = two[1]
          PF = FP / (FP + TN)
          bal5 = 1 - (sqrt(pow(PF, 2) + pow((1 - racall5), 2)) / sqrt(2))
          auc = roc_auc_score(test_target, y_predict5)
          fm = f1_score(test_target, y_predict5, average='macro')
          print("LR_auc:\n", auc)
          print("LR_f-measure:\n", fm)
          print("LR_bal:\n", bal5)
        #bal_sum.append(bal5)
         #   fm_sum.append(fm)
          #  auc_sum.append(auc)
        print("\n")

        if predict_type == 'RF':
            # 5、RF
          estimator6 = RandomForestClassifier(n_estimators=50)
          estimator6.fit(smo_train_data, smo_train_target)
          y_predict6 = estimator6.predict(test_data)
          racall6 = recall_score(test_target, y_predict6, average='macro')
          one, two = confusion_matrix(y_true=test_target, y_pred=y_predict6)
          TP = one[0]
          FN = one[1]
          FP = two[0]
          TN = two[1]
          PF = FP / (FP + TN)
          bal6 = 1 - (sqrt(pow(PF, 2) + pow((1 - racall6), 2)) / sqrt(2))
          auc = roc_auc_score(test_target, y_predict6)
          fm = f1_score(test_target, y_predict6, average='macro')
          print("RF_auc:\n", auc)
          print("RF_f-measure:\n", fm)
          print("RF_bal:\n", bal6)
            #bal_sum.append(bal6)
            #fm_sum.append(fm)
            #auc_sum.append(auc)
            #print("\n")

        if predict_type == 'NB':
            # 6、NB
          estimator4 = GaussianNB()
          estimator4.fit(smo_train_data, smo_train_target)
          y_predict4 = estimator4.predict(test_data)
          racall4 = recall_score(test_target, y_predict4, average='macro')
          one, two = confusion_matrix(y_true=test_target, y_pred=y_predict4)
          TP = one[0]
          FN = one[1]
          FP = two[0]
          TN = two[1]
          PF = FP / (FP + TN)
          bal4 = 1 - (sqrt(pow(PF, 2) + pow((1 - racall4), 2)) / sqrt(2))
          auc = roc_auc_score(test_target, y_predict4)
          fm = f1_score(test_target, y_predict4, average='macro')
          print("NB_auc:\n", auc)
          print("NB_f-measure:\n", fm)
          print("NB_bal:\n", bal4)
          #bal_sum.append(bal4)
         #   fm_sum.append(fm)
          #  auc_sum.append(auc)
          print("\n")

      #print("bal-mean:\n", np.mean(bal_sum))
      #print("fm-mean:\n", np.mean(fm_sum))
      #print("auc-mean:\n", np.mean(auc_sum))



if __name__ == '__main__':
    Bal_sum = []
    bal = 0
    fm_sum = []
    fm = 0
    data = data
    target = target
    type = 'SVM'
    ska = SKA(n_components_num=25)
    ska.predict(type, data, target)

