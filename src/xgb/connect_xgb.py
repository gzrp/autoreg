import numpy as np

from src.data.dataset.connect import get_connect_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

if __name__ == '__main__':
    data_dir = "/data/ruipeng/workdir/autoreg/.data/connect"
    train_set, val_set, test_set = get_connect_dataset(data_dir)

    X_train, y_train = train_set.X, train_set.y
    X_test, y_test = test_set.X, test_set.y
    n_classes = len(np.unique(y_train))
    print("类别数:", n_classes)
    # step1
    # cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
    # ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0,
    #               'subsample': 0.8, 'colsample_bytree': 0.8,
    #               'objective': 'multi:softprob', 'num_class': n_classes,'eval_metric': 'mlogloss'}
    #
    # print("训练模型并选择最优参数......")
    # # 使用5-fold cross-validation来优选最佳的模型
    # optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='balanced_accuracy', cv=5, n_jobs=-1, verbose=10)
    # optimized_GBM.fit(X_train, y_train)
    #
    # print("最佳参数：", optimized_GBM.best_params_)
    #
    # means = optimized_GBM.cv_results_['mean_test_score']
    # stds = optimized_GBM.cv_results_['std_test_score']
    #
    # for mean, std, params in zip(means, stds, optimized_GBM.cv_results_['params']):
    #     print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))
    #
    # y_pred = optimized_GBM.predict(X_test)
    # print(classification_report(y_test, y_pred))
    #
    # # ====== Balanced Accuracy ======
    # bacc = balanced_accuracy_score(y_test, y_pred)
    # print("Balanced Accuracy (BAcc):", bacc)
    # # {'max_depth': 7, 'min_child_weight': 3}

    # cv_params = {'learning_rate': [0.1, 0.05, 0.01], 'subsample': [0.7, 0.8, 0.9]}
    # ind_params = {'max_depth': 7, 'n_estimators': 1000, 'seed': 0, 'min_child_weight': 3, 'colsample_bytree': 0.8,
    #               'objective': 'multi:softprob', 'num_class': n_classes,'eval_metric': 'mlogloss'}
    #
    # print("训练模型并选择最优参数......")
    # # 使用5-fold cross-validation来优选最佳的模型
    # optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='balanced_accuracy', cv=5, n_jobs=-1, verbose=10)
    # optimized_GBM.fit(X_train, y_train)
    #
    # print("最佳参数：", optimized_GBM.best_params_)
    #
    # means = optimized_GBM.cv_results_['mean_test_score']
    # stds = optimized_GBM.cv_results_['std_test_score']
    #
    # for mean, std, params in zip(means, stds, optimized_GBM.cv_results_['params']):
    #     print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

    #最佳参数： 最佳参数：{'learning_rate': 0.1, 'subsample': 0.8}

    ind_params = {'max_depth': 7, 'min_child_weight': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'n_estimators': 3001,
                  'seed': 0, 'colsample_bytree': 0.8, 'objective': 'multi:softprob', 'num_class': n_classes,'eval_metric': 'mlogloss'}
    eval_set = [(X_test, y_test)]

    model = XGBClassifier(**ind_params)
    result = model.fit(X_train, y_train, early_stopping_rounds=200, eval_set=eval_set, verbose=20)
    print("最佳迭代次数:", result.best_iteration)

    y_pred = model.predict(X_test, ntree_limit=result.best_iteration)
    print(classification_report(y_test, y_pred))

    # ====== Balanced Accuracy ======
    bacc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy (BAcc):", bacc)


    # [0]	validation_0-mlogloss:1.06942
    # [20]	validation_0-mlogloss:0.75250
    # [40]	validation_0-mlogloss:0.64059
    # [60]	validation_0-mlogloss:0.58573
    # [80]	validation_0-mlogloss:0.55065
    # [100]	validation_0-mlogloss:0.52645
    # [120]	validation_0-mlogloss:0.50855
    # [140]	validation_0-mlogloss:0.49497
    # [160]	validation_0-mlogloss:0.48252
    # [180]	validation_0-mlogloss:0.47184
    # [200]	validation_0-mlogloss:0.46274
    # [220]	validation_0-mlogloss:0.45462
    # [240]	validation_0-mlogloss:0.44636
    # [260]	validation_0-mlogloss:0.43930
    # [280]	validation_0-mlogloss:0.43322
    # [300]	validation_0-mlogloss:0.42748
    # [320]	validation_0-mlogloss:0.42246
    # [340]	validation_0-mlogloss:0.41654
    # [360]	validation_0-mlogloss:0.41195
    # [380]	validation_0-mlogloss:0.40783
    # [400]	validation_0-mlogloss:0.40360
    # [420]	validation_0-mlogloss:0.39997
    # [440]	validation_0-mlogloss:0.39636
    # [460]	validation_0-mlogloss:0.39339
    # [480]	validation_0-mlogloss:0.39042
    # [500]	validation_0-mlogloss:0.38785
    # [520]	validation_0-mlogloss:0.38437
    # [540]	validation_0-mlogloss:0.38168
    # [560]	validation_0-mlogloss:0.37890
    # [580]	validation_0-mlogloss:0.37610
    # [600]	validation_0-mlogloss:0.37399
    # [620]	validation_0-mlogloss:0.37191
    # [640]	validation_0-mlogloss:0.37014
    # [660]	validation_0-mlogloss:0.36846
    # [680]	validation_0-mlogloss:0.36649
    # [700]	validation_0-mlogloss:0.36470
    # [720]	validation_0-mlogloss:0.36303
    # [740]	validation_0-mlogloss:0.36145
    # [760]	validation_0-mlogloss:0.35947
    # [780]	validation_0-mlogloss:0.35790
    # [800]	validation_0-mlogloss:0.35660
    # [820]	validation_0-mlogloss:0.35528
    # [840]	validation_0-mlogloss:0.35388
    # [860]	validation_0-mlogloss:0.35264
    # [880]	validation_0-mlogloss:0.35167
    # [900]	validation_0-mlogloss:0.35052
    # [920]	validation_0-mlogloss:0.34939
    # [940]	validation_0-mlogloss:0.34835
    # [960]	validation_0-mlogloss:0.34687
    # [980]	validation_0-mlogloss:0.34566
    # [1000]	validation_0-mlogloss:0.34463
    # [1020]	validation_0-mlogloss:0.34402
    # [1040]	validation_0-mlogloss:0.34316
    # [1060]	validation_0-mlogloss:0.34232
    # [1080]	validation_0-mlogloss:0.34139
    # [1100]	validation_0-mlogloss:0.34055
    # [1120]	validation_0-mlogloss:0.33997
    # [1140]	validation_0-mlogloss:0.33923
    # [1160]	validation_0-mlogloss:0.33846
    # [1180]	validation_0-mlogloss:0.33737
    # [1200]	validation_0-mlogloss:0.33670
    # [1220]	validation_0-mlogloss:0.33608
    # [1240]	validation_0-mlogloss:0.33530
    # [1260]	validation_0-mlogloss:0.33453
    # [1280]	validation_0-mlogloss:0.33400
    # [1300]	validation_0-mlogloss:0.33338
    # [1320]	validation_0-mlogloss:0.33279
    # [1340]	validation_0-mlogloss:0.33244
    # [1360]	validation_0-mlogloss:0.33174
    # [1380]	validation_0-mlogloss:0.33127
    # [1400]	validation_0-mlogloss:0.33086
    # [1420]	validation_0-mlogloss:0.33031
    # [1440]	validation_0-mlogloss:0.32978
    # [1460]	validation_0-mlogloss:0.32930
    # [1480]	validation_0-mlogloss:0.32892
    # [1500]	validation_0-mlogloss:0.32864
    # [1520]	validation_0-mlogloss:0.32823
    # [1540]	validation_0-mlogloss:0.32764
    # [1560]	validation_0-mlogloss:0.32728
    # [1580]	validation_0-mlogloss:0.32678
    # [1600]	validation_0-mlogloss:0.32644
    # [1620]	validation_0-mlogloss:0.32629
    # [1640]	validation_0-mlogloss:0.32610
    # [1660]	validation_0-mlogloss:0.32571
    # [1680]	validation_0-mlogloss:0.32537
    # [1700]	validation_0-mlogloss:0.32500
    # [1720]	validation_0-mlogloss:0.32476
    # [1740]	validation_0-mlogloss:0.32441
    # [1760]	validation_0-mlogloss:0.32423
    # [1780]	validation_0-mlogloss:0.32388
    # [1800]	validation_0-mlogloss:0.32365
    # [1820]	validation_0-mlogloss:0.32355
    # [1840]	validation_0-mlogloss:0.32328
    # [1860]	validation_0-mlogloss:0.32309
    # [1880]	validation_0-mlogloss:0.32283
    # [1900]	validation_0-mlogloss:0.32265
    # [1920]	validation_0-mlogloss:0.32251
    # [1940]	validation_0-mlogloss:0.32231
    # [1960]	validation_0-mlogloss:0.32189
    # [1980]	validation_0-mlogloss:0.32181
    # [2000]	validation_0-mlogloss:0.32161
    # [2020]	validation_0-mlogloss:0.32144
    # [2040]	validation_0-mlogloss:0.32134
    # [2060]	validation_0-mlogloss:0.32116
    # [2080]	validation_0-mlogloss:0.32098
    # [2100]	validation_0-mlogloss:0.32071
    # [2120]	validation_0-mlogloss:0.32044
    # [2140]	validation_0-mlogloss:0.32036
    # [2160]	validation_0-mlogloss:0.32023
    # [2180]	validation_0-mlogloss:0.32020
    # [2200]	validation_0-mlogloss:0.32008
    # [2220]	validation_0-mlogloss:0.31990
    # [2240]	validation_0-mlogloss:0.31998
    # [2260]	validation_0-mlogloss:0.31993
    # [2280]	validation_0-mlogloss:0.31982
    # [2300]	validation_0-mlogloss:0.31968
    # [2320]	validation_0-mlogloss:0.31968
    # [2340]	validation_0-mlogloss:0.31956
    # [2360]	validation_0-mlogloss:0.31962
    # [2380]	validation_0-mlogloss:0.31958
    # [2400]	validation_0-mlogloss:0.31946
    # [2420]	validation_0-mlogloss:0.31931
    # [2440]	validation_0-mlogloss:0.31920
    # [2460]	validation_0-mlogloss:0.31907
    # [2480]	validation_0-mlogloss:0.31905
    # [2500]	validation_0-mlogloss:0.31896
    # [2520]	validation_0-mlogloss:0.31899
    # [2540]	validation_0-mlogloss:0.31901
    # [2560]	validation_0-mlogloss:0.31899
    # [2580]	validation_0-mlogloss:0.31898
    # [2600]	validation_0-mlogloss:0.31901
    # [2620]	validation_0-mlogloss:0.31900
    # [2640]	validation_0-mlogloss:0.31889
    # [2660]	validation_0-mlogloss:0.31876
    # [2680]	validation_0-mlogloss:0.31882
    # [2700]	validation_0-mlogloss:0.31876
    # [2720]	validation_0-mlogloss:0.31858
    # [2740]	validation_0-mlogloss:0.31865
    # [2760]	validation_0-mlogloss:0.31864
    # [2780]	validation_0-mlogloss:0.31868
    # [2800]	validation_0-mlogloss:0.31867
    # [2820]	validation_0-mlogloss:0.31860
    # [2840]	validation_0-mlogloss:0.31866
    # [2860]	validation_0-mlogloss:0.31869
    # [2880]	validation_0-mlogloss:0.31866
    # [2900]	validation_0-mlogloss:0.31878
    # [2920]	validation_0-mlogloss:0.31879
    # 最佳迭代次数: 2721
    #
    #               precision    recall  f1-score   support
    #
    #            0       0.84      0.86      0.85      3327
    #            1       0.92      0.95      0.93      8895
    #            2       0.58      0.40      0.47      1290
    #
    #     accuracy                           0.88     13512
    #    macro avg       0.78      0.74      0.75     13512
    # weighted avg       0.87      0.88      0.87     13512
    #
    # Balanced Accuracy (BAcc): 0.7351571755109956