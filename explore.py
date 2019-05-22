import numpy as np
import matplotlib.pyplot as plt



class Explore():
    def __init__(self, config):
        self.config = config

    def check_target_rto(self, data):
        no_shows = np.asarray(data['No-show'])
        no_show_rto = np.sum(no_shows) / len(no_shows)
        show_rto = 1-no_show_rto
        bardata = {'no show' : np.round(no_show_rto, 4), 'show' : np.round(show_rto, 4)}

        plt.bar(range(len(bardata)), list(bardata.values()), align='center', color='lightgreen')
        plt.xticks(range(len(bardata)), list(bardata.keys()))
        plt.title('No Show Ratios')
        plt.savefig('./out/no_sho_rto.png')
        plt.close()

    def draw_boxplots(self, data):
        show_data = data[data['No-show']==0]
        noshow_data = data[data['No-show']==1]
        def bp_wrapper(attr):
            fig, ax = plt.subplots()
            ax.boxplot([show_data[attr], noshow_data[attr]])
            ax.set_title('{} - target dependencies'.format(attr))
            ax.set_xticklabels(labels=['Show', 'No-show'])
            ax.set_ylabel(attr)
            plt.savefig('./out/feature_eda/{}-target.png'.format(attr))
            plt.close()
        bp_wrapper('date_lag')
        bp_wrapper('Age')
        bp_wrapper('humidity')
        bp_wrapper('rain')
        bp_wrapper('solar')
        bp_wrapper('temp')
        bp_wrapper('dewpoint')



    def draw_barplots(self, data) :
        def bar_wrapper(attr, factors):
            factor_dict = {}
            for fac in factors :
                factor_dict[fac] = []
            for i, fac in enumerate(factors) :
                fac_df = data[data[attr]==fac]
                ns = fac_df[fac_df['No-show']==1].shape[0]/fac_df.shape[0]
                factor_dict[fac].append(ns)
            fig, ax = plt.subplots()
            for i, fac in enumerate(factors):
                ax.bar(i, factor_dict[fac], label=fac, width=0.5)
            ax.set_title('{} - target dependencies (%)'.format(attr))
            ax.set_xticklabels(['']*len(factors))
            ax.legend()
            plt.savefig('./out/feature_eda/{}-target.png'.format(attr))
            plt.close()

        bar_wrapper('Gender', [0, 1])
        bar_wrapper('age_bin', [0,1,2,3])
        bar_wrapper('Scholarship', [0,1])
        bar_wrapper('Hipertension', [0,1])
        bar_wrapper('Diabetes', [0,1])
        bar_wrapper('Alcoholism', [0,1])
        bar_wrapper('Handcap', [0,1])
        bar_wrapper('SMS_received', [0,1])
        bar_wrapper('weekday', [0,1,2,3,4,5])
        bar_wrapper('SMS_received', [0,1])
        bar_wrapper('disease_num', [0,1,2,3,4])
        bar_wrapper('rain', [0,1])
        bar_wrapper('season', [0,1])
