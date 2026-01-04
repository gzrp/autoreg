from src.data.dataset.frappe import get_frappe_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score


if __name__ == '__main__':
    data_dir = "/data/ruipeng/workdir/autoreg/.data/frappe"
    train_set, val_set, test_set = get_frappe_dataset(data_dir)

    X_train, y_train = train_set.X, train_set.y
    X_test, y_test = test_set.X, test_set.y
    print("加载完数据~")
    # step1
    # cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
    # ind_params = {'learning_rate': 0.1, 'n_estimators': 300, 'seed': 0,
    #               'subsample': 0.8, 'colsample_bytree': 0.8,
    #               'objective': 'binary:logistic'}
    #
    # print("训练模型并选择最优参数......")
    # # 使用5-fold cross-validation来优选最佳的模型
    # optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=8, verbose=10)
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
    # # 获取正类的预测概率
    # y_score = optimized_GBM.predict_proba(X_test)[:, 1]
    # auc = roc_auc_score(y_test, y_score)
    # print("AUC:", auc)

    # #  {'max_depth': 7, 'min_child_weight': 1}

    # cv_params = {'learning_rate': [0.1, 0.05, 0.01], 'subsample': [0.7, 0.8, 0.9]}
    # ind_params = {'max_depth': 7, 'n_estimators': 300, 'seed': 0, 'min_child_weight': 1, 'colsample_bytree': 0.8,
    #               'objective': 'binary:logistic'}
    #
    # print("训练模型并选择最优参数......")
    # # 使用5-fold cross-validation来优选最佳的模型
    # optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=12,
    #                              verbose=10)
    # optimized_GBM.fit(X_train, y_train)
    #
    # print("最佳参数：", optimized_GBM.best_params_)
    #
    # means = optimized_GBM.cv_results_['mean_test_score']
    # stds = optimized_GBM.cv_results_['std_test_score']
    #
    # for mean, std, params in zip(means, stds, optimized_GBM.cv_results_['params']):
    #     print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

    #最佳参数： {'learning_rate': 0.1, 'subsample': 0.9}

    ind_params = {'max_depth': 7, 'min_child_weight': 1, 'learning_rate': 0.1, 'subsample': 0.9, 'n_estimators': 5001,
                  'seed': 0, 'colsample_bytree': 0.8, 'objective': 'binary:logistic'}
    eval_set = [(X_test, y_test)]

    model = XGBClassifier(**ind_params)
    result = model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=eval_set, verbose=50)
    print("最佳迭代次数:", result.best_iteration)

    y_pred = model.predict(X_test, ntree_limit=result.best_iteration)
    print(classification_report(y_test, y_pred))

    # # 获取正类的预测概率
    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    print("AUC:", auc)


    # [0]	validation_0-auc:0.63242
    # [50]	validation_0-auc:0.79406
    # [100]	validation_0-auc:0.83320
    # [150]	validation_0-auc:0.85496
    # [200]	validation_0-auc:0.86770
    # [250]	validation_0-auc:0.87912
    # [300]	validation_0-auc:0.89245
    # [350]	validation_0-auc:0.90384
    # [400]	validation_0-auc:0.90947
    # [450]	validation_0-auc:0.91605
    # [500]	validation_0-auc:0.92069
    # [550]	validation_0-auc:0.92370
    # [600]	validation_0-auc:0.92684
    # [650]	validation_0-auc:0.92967
    # [700]	validation_0-auc:0.93296
    # [750]	validation_0-auc:0.93495
    # [800]	validation_0-auc:0.93591
    # [850]	validation_0-auc:0.93787
    # [900]	validation_0-auc:0.93926
    # [950]	validation_0-auc:0.94119
    # [1000]	validation_0-auc:0.94276
    # [1050]	validation_0-auc:0.94464
    # [1100]	validation_0-auc:0.94579
    # [1150]	validation_0-auc:0.94704
    # [1200]	validation_0-auc:0.94767
    # [1250]	validation_0-auc:0.94852
    # [1300]	validation_0-auc:0.94937
    # [1350]	validation_0-auc:0.95007
    # [1400]	validation_0-auc:0.95060
    # [1450]	validation_0-auc:0.95114
    # [1500]	validation_0-auc:0.95184
    # [1550]	validation_0-auc:0.95242
    # [1600]	validation_0-auc:0.95275
    # [1650]	validation_0-auc:0.95322
    # [1700]	validation_0-auc:0.95363
    # [1750]	validation_0-auc:0.95399
    # [1800]	validation_0-auc:0.95432
    # [1850]	validation_0-auc:0.95437
    # [1900]	validation_0-auc:0.95483
    # [1950]	validation_0-auc:0.95516
    # [2000]	validation_0-auc:0.95554
    # [2050]	validation_0-auc:0.95581
    # [2100]	validation_0-auc:0.95610
    # [2150]	validation_0-auc:0.95651
    # [2200]	validation_0-auc:0.95681
    # [2250]	validation_0-auc:0.95715
    # [2300]	validation_0-auc:0.95753
    # [2350]	validation_0-auc:0.95801
    # [2400]	validation_0-auc:0.95836
    # [2450]	validation_0-auc:0.95870
    # [2500]	validation_0-auc:0.95896
    # [2550]	validation_0-auc:0.95917
    # [2600]	validation_0-auc:0.95944
    # [2650]	validation_0-auc:0.95978
    # [2700]	validation_0-auc:0.96005
    # [2750]	validation_0-auc:0.96024
    # [2800]	validation_0-auc:0.96040
    # [2850]	validation_0-auc:0.96084
    # [2900]	validation_0-auc:0.96100
    # [2950]	validation_0-auc:0.96121
    # [3000]	validation_0-auc:0.96138
    # [3050]	validation_0-auc:0.96168
    # [3100]	validation_0-auc:0.96174
    # [3150]	validation_0-auc:0.96193
    # [3200]	validation_0-auc:0.96210
    # [3250]	validation_0-auc:0.96231
    # [3300]	validation_0-auc:0.96242
    # [3350]	validation_0-auc:0.96256
    # [3400]	validation_0-auc:0.96273
    # [3450]	validation_0-auc:0.96285
    # [3500]	validation_0-auc:0.96304
    # [3550]	validation_0-auc:0.96321
    # [3600]	validation_0-auc:0.96331
    # [3650]	validation_0-auc:0.96341
    # [3700]	validation_0-auc:0.96358
    # [3750]	validation_0-auc:0.96364
    # [3800]	validation_0-auc:0.96381
    # [3850]	validation_0-auc:0.96389
    # [3900]	validation_0-auc:0.96401
    # [3950]	validation_0-auc:0.96409
    # [4000]	validation_0-auc:0.96423
    # [4050]	validation_0-auc:0.96433
    # [4100]	validation_0-auc:0.96446
    # [4150]	validation_0-auc:0.96454
    # [4200]	validation_0-auc:0.96465
    # [4250]	validation_0-auc:0.96471
    # [4300]	validation_0-auc:0.96480
    # [4350]	validation_0-auc:0.96488
    # [4400]	validation_0-auc:0.96496
    # [4450]	validation_0-auc:0.96503
    # [4500]	validation_0-auc:0.96510
    # [4550]	validation_0-auc:0.96521
    # [4600]	validation_0-auc:0.96524
    # [4650]	validation_0-auc:0.96531
    # [4700]	validation_0-auc:0.96538
    # [4750]	validation_0-auc:0.96546
    # [4800]	validation_0-auc:0.96553
    # [4850]	validation_0-auc:0.96561
    # [4900]	validation_0-auc:0.96565
    # [4950]	validation_0-auc:0.96573
    # [5000]	validation_0-auc:0.96575
    # 最佳迭代次数: 4996
    # /data/ruipeng/miniconda3/envs/py312/lib/python3.12/site-packages/xgboost/core.py:91: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    #   warnings.warn(
    #               precision    recall  f1-score   support
    #
    #            0       0.93      0.98      0.95     19324
    #            1       0.95      0.85      0.89      9536
    #
    #     accuracy                           0.93     28860
    #    macro avg       0.94      0.91      0.92     28860
    # weighted avg       0.93      0.93      0.93     28860
    #
    # AUC: 0.9657743740310063