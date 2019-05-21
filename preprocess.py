import pandas as pd
import numpy as np

class Preprocess():
    def __init__(self, config):
        self.config = config

    def split_data(self, data, seed=None):
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle data
        show_data = data[data['No-show']==0]
        noshow_data = data[data['No-show']==1]
        show_bound = int(show_data.shape[0] * self.config.train_ratio / 100)
        noshow_bound = int(noshow_data.shape[0] * self.config.train_ratio / 100)

        if not self.config.imbalanced_target :
            train = pd.concat([show_data.iloc[:show_bound, :].sample(noshow_bound, random_state=seed),
                               noshow_data.iloc[:noshow_bound, :]])
            test = pd.concat([show_data.iloc[show_bound:, :].sample(noshow_data.shape[0]-noshow_bound, random_state=seed),
                              noshow_data.iloc[noshow_bound:, :]])
            train = train.sample(frac=1, random_state=seed).reset_index(drop=True)  # reshuffle data
            test = test.sample(frac=1, random_state=seed).reset_index(drop=True)  # reshuffle data
            return train, test

        train = pd.concat([show_data.iloc[:show_bound, :], noshow_data.iloc[:noshow_bound, :]])
        test = pd.concat([show_data.iloc[show_bound:, :], noshow_data.iloc[noshow_bound:, :]])
        train = train.sample(frac=1, random_state=seed).reset_index(drop=True)  # reshuffle data
        test = test.sample(frac=1, random_state=seed).reset_index(drop=True)  # reshuffle data

        return train, test


    def preprocess(self) :
        data = pd.read_csv('./data/final.csv')
        data = data[data.humidity != 0]
        data = data.drop(['year','month','day','Neighbourhood'], axis=1)
        data['disease_num'] = np.sum([data.Hipertension, data.Diabetes, data.Alcoholism, data.Handcap],axis=0)
        data.loc[data.rain != 0, 'rain'] = 1
        data = data.reset_index().iloc[:,1:]
        data.to_csv('./data/final_prepro.csv',index=False)

        data = self.onehot_encode(data)
        data.to_csv('./data/final_prepro_onehot.csv',index=False)

    def onehot_encode(self, data):
        def encode(col) :
            num = np.unique(col, axis=0)
            num = num.shape[0]
            encoding = pd.DataFrame(np.eye(num)[col])
            return encoding

        agebin = data.age_bin
        agebin = encode(agebin)
        agebin.columns = ['kid','youth','adult','senior']
        data = pd.concat([data, agebin], axis=1)
        data = data.drop(['age_bin'], axis=1)
        '''
        season = data.season
        season = encode(season)
        season.columns = ['spring','summer']
        data = pd.concat([data, season], axis=1)
        data = data.drop(['season'], axis=1)
        '''

        wkday = data.weekday
        wkday = encode(wkday)
        wkday.columns = ['mon','tue','wed','thu','fri','sat']
        data = pd.concat([data, wkday], axis=1)
        data = data.drop(['weekday'], axis=1)

        return data

