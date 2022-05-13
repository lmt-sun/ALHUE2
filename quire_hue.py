from utils import evaluate
from alipy.data_manipulate import split
from alipy.query_strategy.query_labels import QueryInstanceUncertainty, _get_proba_pred, QueryInstanceQBC, \
    QueryInstanceQUIRE
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from pandas.core.frame import DataFrame



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
    def predict(self, data, target):
        #训练集、测试集的划分
        train, test, labled, unlab = split(X=data, y=target, test_ratio=0.3, initial_label_rate=0.1,
                                           split_count=1, all_class=True, saving_path='..')
        unlab = unlab[0].tolist()
        labled = labled[0].tolist()


        # 基于代表性和信息性的的的查询策略QUIRE
        for i in range(94):
          Q = QueryInstanceQUIRE(X=data, y=target, train_idx=train[0])
          select = Q.select(labled, unlab)
          labled.append(select[0])
          unlab.remove(select[0])

        train_data = data.loc[labled]
        train_target = target.loc[labled]
        train_data = DataFrame(train_data)
        train_target = DataFrame(train_target)
        test_data = data.loc[test[0]]
        test_target = target.loc[test[0]]

        # 类不平衡学习方法HUE
        for method in [
            'random',
            'linearity',
            'negexp',
            'reciprocal',
             'limit'
        ]:
            evaluate(
                "Method: {}".format( method.title()),
                #DecisionTreeClassifier(criterion="entropy"),
                #KNeighborsClassifier(n_neighbors=3),
                LinearSVC(random_state=28),
                #LogisticRegression(),
                #RandomForestClassifier(n_estimators=10),
                #GaussianNB(),
                train_data,
                train_target,
                test_data,
                test_target,
                k=5,
                verbose=True,
                sampling=method
            )
        print("*" * 50)



if __name__ == '__main__':
    Bal_sum = []
    bal = 0
    fm_sum = []
    fm = 0
    data = data
    target = target
    for i in range(10):
     ska = SKA()
     ska.predict(data, target)

