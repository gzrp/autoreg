import numpy as np

from src.data.dataset.dionis import get_dionis_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

if __name__ == '__main__':
    data_dir = "/data/ruipeng/workdir/autoreg/.data/dionis"
    train_set, val_set, test_set = get_dionis_dataset(data_dir)

    X_train, y_train = train_set.X, train_set.y
    X_test, y_test = test_set.X, test_set.y
    n_classes = len(np.unique(y_train))
    print("类别数:", n_classes)
    # step1
    # cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
    # ind_params = {'learning_rate': 0.1, 'n_estimators': 300, 'seed': 0,
    #               'subsample': 0.8, 'colsample_bytree': 0.8,
    #               'objective': 'multi:softprob', 'num_class': n_classes,'eval_metric': 'mlogloss'}
    #
    # print("训练模型并选择最优参数......")
    # # 使用5-fold cross-validation来优选最佳的模型
    # optimized_GBM = GridSearchCV(XGBClassifier(**ind_params), cv_params, scoring='accuracy', cv=5, n_jobs=12, verbose=10)
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

    # #  max_depth=5, min_child_weight=3

    # cv_params = {'learning_rate': [0.1, 0.05, 0.01], 'subsample': [0.7, 0.8, 0.9]}
    # ind_params = {'max_depth': 5, 'n_estimators': 100, 'seed': 0, 'min_child_weight': 3, 'colsample_bytree': 0.8,
    #               'objective': 'multi:softprob', 'num_class': n_classes,'eval_metric': 'mlogloss'}
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

    #最佳参数： 最佳参数：{'learning_rate': 0.1, 'subsample': 0.7}

    ind_params = {'max_depth': 5, 'min_child_weight': 3, 'learning_rate': 0.1, 'subsample': 0.7, 'n_estimators': 1000,
                  'seed': 0, 'colsample_bytree': 0.8, 'objective': 'multi:softprob', 'num_class': n_classes,'eval_metric': 'mlogloss'}
    eval_set = [(X_test, y_test)]

    model = XGBClassifier(**ind_params)
    result = model.fit(X_train, y_train, early_stopping_rounds=100, eval_set=eval_set, verbose=20)
    print("最佳迭代次数:", result.best_iteration)

    y_pred = model.predict(X_test, ntree_limit=result.best_iteration)
    print(classification_report(y_test, y_pred))

    # ====== Balanced Accuracy ======
    bacc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy (BAcc):", bacc)


    #