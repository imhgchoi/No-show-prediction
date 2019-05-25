import argparse


def get_args():
    argp = argparse.ArgumentParser(description='No show Prediction for Hospitals',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argp.add_argument('--resolve_data', action='store_true', default=False)
    argp.add_argument('--train', action='store_true', default=False)
    argp.add_argument('--explore', action='store_true', default=False)
    argp.add_argument('--preprocess', action='store_true', default=False)
    argp.add_argument('--train_ratio', type=int, default=70)
    argp.add_argument('--imbalanced_target', action='store_true', default=False)
    argp.add_argument('--natural_test', action='store_true', default=False)
    argp.add_argument('--impt_only', action='store_true', default=False)
    argp.add_argument('--model_type', type=str, default=None,
                      choices=['ensemble','decision_tree','random_forest','extra_trees','logistic_regression',
                               'gradient_boosting','ada_boosting','bagging','multinomial_nb','gaussian_nb','dnn','mlp',
                               'knn','svm'])

    argp.add_argument('--ensemble_models', nargs='+', type=str)
    argp.add_argument('--ensemble_method', type=str, default='vote',
                      choices=['vote','mlp'])

    # tree hyperparameters
    argp.add_argument('--max_depth', type=int, default=8) # optimal : dt - 5 / rf - 11 / et - 8
    argp.add_argument('--min_samples_split', type=int, default=50) # optimal : dt - 500 / rf - 300 / et - 50
    argp.add_argument('--min_samples_leaf', type=int, default=10) # optimal : dt - 500 / rf - 500 / et - 10
    argp.add_argument('--max_features', type=int, default=None)

    # random forest / extra trees hyperparemeters
    argp.add_argument('--n_estimators', type=int, default=500) # optimal : rf - 500 / et - 500


    # mlp hyperparameters
    argp.add_argument('--lr', type=float, default=0.02)
    argp.add_argument('--max_epoch', type=int, default=500) # optimal : mlp - 3000 /
    argp.add_argument('--output_activation', type=str, default='sigmoid', choices=['sigmoid','softmax'])
    argp.add_argument('--disable_early_stop', action='store_true', default=False)
    argp.add_argument('--early_stop_std', type=float, default=0.000001) # optimal : mlp - 0.000001 / dnn -

    # KNN hyperparameter
    argp.add_argument('--neighbors', type=int, default=19) # optimal : 19

    # SVM hyperparameter
    argp.add_argument('--C', type=float, default=10.0) # sub-optimal : above 5.0

    return argp.parse_args()
