from config import get_args
from data_downloader import DataDownloader
from explore import Explore
from preprocess import Preprocess
from train import Train
import numpy as np


def main():
    config = get_args()
    if config.train and config.model_type == None :
        print('You need go specify a model type')
        exit()
    else :
        data = DataDownloader(config)
    eda = Explore(config)
    prepro = Preprocess(config)
    trainer = Train(config)

    # basic wrangling with data
    if config.resolve_data :
        data.get_weather_loc()
        data.create_keys()
        data.merge_tables()

    # preprocessing
    if config.preprocess :
        print('starting preprocess')
        prepro.preprocess()

    data, data_onehot = data.load_data()

    # EDA
    if config.explore :
        print('starting EDA')
        eda.check_target_rto(data)
        eda.draw_hist(data)
        eda.draw_boxplots(data)
        eda.draw_barplots(data)

    # train and test
    if config.train :
        print('starting training')
        seed = np.random.randint(0,9999)
        train, test = prepro.split_data(data, seed=seed)
        train_oh, test_oh = prepro.split_data(data_onehot, seed=seed)
        if config.ensemble_models == None or len(config.ensemble_models) == 1 :
            if config.model_type == 'decision_tree':
                trainer.decision_tree(train,test)
            elif config.model_type == 'random_forest' :
                trainer.random_forest(train, test)
            elif config.model_type == 'extra_trees' :
                trainer.extra_trees(train, test)
            elif config.model_type == 'logistic_regression' :
                trainer.logistic_regression(train_oh,test_oh)
            elif config.model_type == 'gradient_boosting' :
                trainer.boosting(train, test, ada=False)
            elif config.model_type == 'ada_boosting' :
                trainer.boosting(train, test, ada=True)
            elif config.model_type == 'bagging' :
                trainer.bagging(train, test)
            elif config.model_type == 'dnn' :
                trainer.neural_net(train_oh,test_oh, config.model_type)
            elif config.model_type == 'mlp' :
                trainer.neural_net(train_oh,test_oh, config.model_type)
            elif config.model_type == 'multinomial_nb':
                if config.imbalanced_target :
                    trainer.complement_nb(train_oh, test_oh)
                else :
                    trainer.multinomial_nb(train_oh, test_oh)
            elif config.model_type == 'gaussian_nb':
                trainer.gaussian_nb(train_oh, test_oh)
            elif config.model_type == 'knn' :
                trainer.K_nearneighbor(train_oh, test_oh)
            elif config.model_type == 'svm' :
                trainer.svm(train_oh, test_oh)

        else :
            models = []
            models_oh = []
            models_torch = []
            if 'decision_tree' in config.ensemble_models :
                models.append(trainer.decision_tree(train, test))
            if 'random_forest' in config.ensemble_models :
                models.append(trainer.random_forest(train, test))
            if 'extra_trees' in config.ensemble_models :
                models.append(trainer.extra_trees(train, test))
            if 'logistic_regression' in config.ensemble_models :
                models_oh.append(trainer.logistic_regression(train_oh,test_oh))
            if 'gradient_boosting' in config.ensemble_models :
                models.append(trainer.boosting(train, test, ada=False))
            if 'ada_boosting' in config.ensemble_models :
                models.append(trainer.boosting(train, test, ada=True))
            if 'bagging' in config.ensemble_models :
                models.append(trainer.bagging(train, test))
            if 'dnn' in config.ensemble_models :
                models_torch.append(trainer.neural_net(train_oh, test_oh, 'dnn'))
            if 'mlp' in config.ensemble_models :
                models_torch.append(trainer.neural_net(train_oh, test_oh, 'mlp'))
            if 'multinomial_nb' in config.ensemble_models :
                if config.imbalanced_target :
                    models_oh.append(trainer.complement_nb(train_oh, test_oh))
                else :
                    models_oh.append(trainer.multinomial_nb(train_oh, test_oh))
            if 'gaussian_nb' in config.ensemble_models :
                models_oh.append(trainer.gaussian_nb(train_oh, test_oh))
            if 'knn' in config.ensemble_models :
                models_oh.append(trainer.K_nearneighbor(train_oh, test_oh))
            if 'svm' in config.ensemble_models :
                models.append(trainer.svm(train_oh, test_oh))

            trainer.ensemble(models, models_oh, models_torch, train, test, train_oh, test_oh)



if __name__ == '__main__' :
    main()