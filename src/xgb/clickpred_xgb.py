from src.data.dataset.clickpred import get_clickpred_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score


if __name__ == '__main__':
    data_dir = "/data/ruipeng/workdir/autoreg/.data/clickpred"
    train_set, val_set, test_set = get_clickpred_dataset(data_dir)

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
    #
    # # {'max_depth': 5, 'min_child_weight': 1}

    # cv_params = {'learning_rate': [0.1, 0.05, 0.01], 'subsample': [0.7, 0.8, 0.9]}
    # ind_params = {'max_depth': 5, 'n_estimators': 1000, 'seed': 0, 'min_child_weight': 1, 'colsample_bytree': 0.8,
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

    #最佳参数： 最佳参数： {'learning_rate': 0.01, 'subsample': 0.8}

    ind_params = {'max_depth': 5, 'min_child_weight': 1, 'learning_rate': 0.01, 'subsample': 0.8, 'n_estimators': 101,
                  'seed': 0, 'colsample_bytree': 0.8, 'objective': 'binary:logistic'}
    eval_set = [(X_test, y_test)]

    model = XGBClassifier(**ind_params)
    result = model.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=20)
    print("最佳迭代次数:", result.best_iteration)

    y_pred = model.predict(X_test, ntree_limit=result.best_iteration)
    print(classification_report(y_test, y_pred))

    # # 获取正类的预测概率
    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    print("AUC:", auc)


    # [0]	validation_0-auc:0.66490
    # [20]	validation_0-auc:0.68876
    # [40]	validation_0-auc:0.69234
    # [60]	validation_0-auc:0.69295
    # [80]	validation_0-auc:0.69496
    # [100]	validation_0-auc:0.69618
    #
    #
    #               precision    recall  f1-score   support
    #
    #            0       0.84      1.00      0.91     66479
    #            1       0.74      0.03      0.06     13418
    #
    #     accuracy                           0.84     79897
    #    macro avg       0.79      0.51      0.48     79897
    # weighted avg       0.82      0.84      0.77     79897
    #
    # AUC: 0.6961958682808218