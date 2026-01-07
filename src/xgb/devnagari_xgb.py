import numpy as np

from src.data.dataset.devnagari import get_devnagari_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

if __name__ == '__main__':
    data_dir = "/data/ruipeng/workdir/autoreg/.data/devnagari"
    train_set, val_set, test_set = get_devnagari_dataset(data_dir)

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

    # #  {'max_depth': 5, 'min_child_weight': 1}

    # cv_params = {'learning_rate': [0.1, 0.05, 0.01], 'subsample': [0.7, 0.8, 0.9]}
    # ind_params = {'max_depth': 5, 'n_estimators': 300, 'seed': 0, 'min_child_weight': 1, 'colsample_bytree': 0.8,
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

    #最佳参数： 最佳参数：{'learning_rate': 0.1, 'subsample': 0.7}

    ind_params = {'max_depth': 5, 'min_child_weight': 1, 'learning_rate': 0.1, 'subsample': 0.7, 'n_estimators': 1000,
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


    # [0]	validation_0-mlogloss:3.23895
    # [20]	validation_0-mlogloss:1.43288
    # [40]	validation_0-mlogloss:0.95925
    # [60]	validation_0-mlogloss:0.72795
    # [80]	validation_0-mlogloss:0.59223
    # [100]	validation_0-mlogloss:0.50185
    # [120]	validation_0-mlogloss:0.43903
    # [140]	validation_0-mlogloss:0.39230
    # [160]	validation_0-mlogloss:0.35642
    # [180]	validation_0-mlogloss:0.32872
    # [200]	validation_0-mlogloss:0.30668
    # [220]	validation_0-mlogloss:0.28847
    # [240]	validation_0-mlogloss:0.27342
    # [260]	validation_0-mlogloss:0.26126
    # [280]	validation_0-mlogloss:0.25121
    # [300]	validation_0-mlogloss:0.24293
    # [320]	validation_0-mlogloss:0.23606
    # [340]	validation_0-mlogloss:0.22985
    # [360]	validation_0-mlogloss:0.22466
    # [380]	validation_0-mlogloss:0.22038
    # [400]	validation_0-mlogloss:0.21665
    # [420]	validation_0-mlogloss:0.21360
    # [440]	validation_0-mlogloss:0.21126
    # [460]	validation_0-mlogloss:0.20896
    # [480]	validation_0-mlogloss:0.20686
    # [500]	validation_0-mlogloss:0.20498
    # [520]	validation_0-mlogloss:0.20338
    # [540]	validation_0-mlogloss:0.20198
    # [560]	validation_0-mlogloss:0.20088
    # [580]	validation_0-mlogloss:0.19968
    # [600]	validation_0-mlogloss:0.19886
    # [620]	validation_0-mlogloss:0.19791
    # [640]	validation_0-mlogloss:0.19724
    # [660]	validation_0-mlogloss:0.19651
    # [680]	validation_0-mlogloss:0.19581
    # [700]	validation_0-mlogloss:0.19518
    # [720]	validation_0-mlogloss:0.19476
    # [740]	validation_0-mlogloss:0.19435
    # [760]	validation_0-mlogloss:0.19393
    # [780]	validation_0-mlogloss:0.19339
    # [800]	validation_0-mlogloss:0.19304
    # [820]	validation_0-mlogloss:0.19261
    # [840]	validation_0-mlogloss:0.19234
    # [860]	validation_0-mlogloss:0.19220
    # [880]	validation_0-mlogloss:0.19202
    # [900]	validation_0-mlogloss:0.19190
    # [920]	validation_0-mlogloss:0.19168
    # [940]	validation_0-mlogloss:0.19156
    # [960]	validation_0-mlogloss:0.19137
    # [980]	validation_0-mlogloss:0.19117
    # [999]	validation_0-mlogloss:0.19113
    # 最佳迭代次数: 998
    # /data/ruipeng/miniconda3/envs/py312/lib/python3.12/site-packages/xgboost/core.py:91: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.
    #   warnings.warn(
    #               precision    recall  f1-score   support
    #
    #            0       0.96      0.97      0.96       400
    #            1       0.94      0.94      0.94       400
    #            2       0.95      0.94      0.94       400
    #            3       0.90      0.89      0.90       400
    #            4       0.93      0.91      0.92       400
    #            5       0.93      0.96      0.94       400
    #            6       0.94      0.91      0.92       400
    #            7       0.97      0.96      0.97       400
    #            8       0.95      0.96      0.96       400
    #            9       0.96      0.95      0.96       400
    #           10       0.95      0.95      0.95       400
    #           11       0.95      0.97      0.96       400
    #           12       0.93      0.94      0.93       400
    #           13       0.94      0.94      0.94       400
    #           14       0.95      0.97      0.96       400
    #           15       0.96      0.96      0.96       400
    #           16       0.90      0.91      0.90       400
    #           17       0.92      0.91      0.92       400
    #           18       0.94      0.91      0.93       400
    #           19       0.94      0.93      0.93       400
    #           20       0.89      0.94      0.92       400
    #           21       0.96      0.95      0.96       400
    #           22       0.94      0.90      0.92       400
    #           23       0.94      0.91      0.93       400
    #           24       0.93      0.94      0.94       400
    #           25       0.88      0.90      0.89       400
    #           26       0.95      0.95      0.95       400
    #           27       0.99      0.96      0.97       400
    #           28       0.91      0.94      0.93       400
    #           29       0.93      0.96      0.94       400
    #           30       0.92      0.95      0.93       400
    #           31       0.94      0.92      0.93       400
    #           32       0.96      0.93      0.94       400
    #           33       0.94      0.95      0.95       400
    #           34       0.95      0.94      0.94       400
    #           35       0.94      0.95      0.95       400
    #           36       0.98      0.99      0.99       400
    #           37       0.98      0.98      0.98       400
    #           38       0.94      0.96      0.95       400
    #           39       0.98      0.94      0.96       400
    #           40       0.98      0.97      0.98       400
    #           41       0.97      0.97      0.97       400
    #           42       0.96      0.96      0.96       400
    #           43       0.98      0.97      0.98       400
    #           44       0.97      0.97      0.97       400
    #           45       0.98      0.98      0.98       400
    #
    #     accuracy                           0.95     18400
    #    macro avg       0.95      0.95      0.95     18400
    # weighted avg       0.95      0.95      0.95     18400
    #
    # Balanced Accuracy (BAcc): 0.9453260869565215