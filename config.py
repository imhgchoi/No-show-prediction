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
    argp.add_argument('--model_type', type=str, default=None,
                      choices=['ensemble','decision_tree','random_forest','extra_trees','logistic_regression',
                               'gradient_boosting','ada_boosting','bagging','multinomial_nb','gaussian_nb','mlp'])

    argp.add_argument('--ensemble_models', nargs='+', type=str)
    argp.add_argument('--ensemble_method', type=str, default='vote',
                      choices=['vote','mlp'])

    # tree hyperparameters
    argp.add_argument('--max_depth', type=int, default=5)
    argp.add_argument('--min_samples_split', type=int, default=5)
    argp.add_argument('--min_samples_leaf', type=int, default=1)
    argp.add_argument('--max_features', type=int, default=None)


    # random forest / extra trees hyperparemeters
    argp.add_argument('--n_estimators', type=int, default=100)

    # mlp hyperparameters
    argp.add_argument('--lr', type=float, default=0.02)
    argp.add_argument('--max_epoch', type=int, default=1000)
    argp.add_argument('--output_activation', type=str, default='sigmoid', choices=['sigmoid','softmax'])
    argp.add_argument('--early_stop_std', type=float, default=0.0002)




    return argp.parse_args()
