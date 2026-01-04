from src.data.dataset.ccfraud import get_ccfraud_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score


if __name__ == '__main__':
    data_dir = "/data/ruipeng/workdir/autoreg/.data/ccfraud"
    train_set, val_set, test_set = get_ccfraud_dataset(data_dir)

    X_train, y_train = train_set.X, train_set.y
    X_test, y_test = test_set.X, test_set.y

    # step1
    # cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
    # ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0,
    #               'subsample': 0.8, 'colsample_bytree': 0.8,
    #               'objective': 'binary:logistic'}
    #
    # print("训练模型并选择最优参数......")
    # # 使用5-fold cross-validation来优选最佳的模型
    # optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=-1, verbose=10)
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

    # 最佳参数{'max_depth': 5, 'min_child_weight': 3}

    # cv_params = {'learning_rate': [0.1, 0.05, 0.01], 'subsample': [0.7, 0.8, 0.9]}
    # ind_params = {'max_depth': 5, 'n_estimators': 1000, 'seed': 0, 'min_child_weight': 3, 'colsample_bytree': 0.8,
    #               'objective': 'binary:logistic'}
    #
    # print("训练模型并选择最优参数......")
    # # 使用5-fold cross-validation来优选最佳的模型
    # optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=-1,
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

    #最佳参数： 最佳参数： {'learning_rate': 0.1, 'subsample': 0.8}

    ind_params = {'max_depth': 5, 'min_child_weight': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'n_estimators': 1000,
                  'seed': 0, 'colsample_bytree': 0.8, 'objective': 'binary:logistic'}
    eval_set = [(X_test, y_test)]

    model = XGBClassifier(**ind_params)
    result = model.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=eval_set, verbose=20)
    print("最佳迭代次数:", result.best_iteration)

    y_pred = model.predict(X_test, ntree_limit=result.best_iteration)
    print(classification_report(y_test, y_pred))

    # # 获取正类的预测概率
    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    print("AUC:", auc)


    # [0]	validation_0-auc:0.90785
    # [20]	validation_0-auc:0.94342
    # [40]	validation_0-auc:0.96635
    # [60]	validation_0-auc:0.97276
    # [80]	validation_0-auc:0.97806
    # [100]	validation_0-auc:0.97642
    # [120]	validation_0-auc:0.97618
    # [140]	validation_0-auc:0.97935
    # [160]	validation_0-auc:0.97795
    # [180]	validation_0-auc:0.97758
    # [200]	validation_0-auc:0.97883
    # [220]	validation_0-auc:0.97915
    # [240]	validation_0-auc:0.98017
    # [260]	validation_0-auc:0.98042
    # [280]	validation_0-auc:0.97979
    # [300]	validation_0-auc:0.97924
    # [320]	validation_0-auc:0.97931
    # [340]	validation_0-auc:0.97941
    # [351]	validation_0-auc:0.97903
    #
    #               precision    recall  f1-score   support
    #
    #            0       1.00      1.00      1.00     56864
    #            1       0.90      0.81      0.85        98
    #
    #     accuracy                           1.00     56962
    #    macro avg       0.95      0.90      0.92     56962
    # weighted avg       1.00      1.00      1.00     56962
    #
    # AUC: 0.9811700742480447