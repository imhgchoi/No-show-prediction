import datetime
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import urllib
import certifi


def uo(args, **kwargs):
    return urllib.request.urlopen(args, cafile=certifi.where(), **kwargs)

class DataDownloader():
    def __init__(self, config):
        self.config = config


    def get_weather_loc(self):
        # gbrd : solar radiation / prcp : precipication
        weather = pd.read_csv("./data/sudeste.csv", usecols=['wsnm','lat','lon','city','prov','yr','mo','da','hr',
                                                             'temp','dewp','hmdy','gbrd','prcp'])
        weather = weather[weather.yr == 2016]
        weather = weather[((weather.mo == 4) | (weather.mo == 5) | (weather.mo == 6))]
        weather.to_csv("./data/sudeste_relevant.csv")
        weather_locs = weather.drop_duplicates('wsnm')
        weather_locs[['wsnm','city','prov','lat','lon']].to_csv('./data/weather_checker_loc.csv', index=False)

        # get location of hospitals
        loc_dict = {'location':[], 'latitude':[], 'longitude':[]}
        weather = pd.read_csv("./data/no_show.csv")
        weather = weather.drop_duplicates('Neighbourhood')
        hospital_locs = weather.Neighbourhood
        geolocator = Nominatim(timeout=10)
        geolocator.urlopen = uo
        for locs in hospital_locs :
            loc_dict['location'].append(locs)
            if locs == 'BELO HOR. (PAMPULHA)':
                loc = geolocator.geocode('PAMPULHA')
            else :
                loc = geolocator.geocode(locs)
            if loc == None :
                print(locs, 'None')
                loc_dict['latitude'].append(np.nan)
                loc_dict['longitude'].append(np.nan)
            else :
                print(locs, loc.latitude, loc.longitude)
                loc_dict['latitude'].append(loc.latitude)
                loc_dict['longitude'].append(loc.longitude)
        loc_df = pd.DataFrame(loc_dict)
        loc_df = loc_df.fillna(loc_df.mean())
        loc_df.to_csv('./data/hospital_loc.csv', index=False)

    def create_keys(self):
        hos_loc = pd.read_csv('./data/hospital_loc.csv')
        wea_loc = pd.read_csv('./data/weather_checker_loc.csv')

        adjacent_inst = []
        for _, hl_row in hos_loc.iterrows():
            hlat = hl_row['latitude']
            hlon = hl_row['longitude']
            min_wsnm = None
            min_value = 99999
            for idx, wl_row in wea_loc.iterrows():
                wlat = wl_row['lat']
                wlon = wl_row['lon']
                dist = np.sqrt((hlat-wlat)**2 + (hlon-wlon)**2)
                if dist < min_value :
                    min_wsnm = wl_row['wsnm']
                    min_value = dist
            adjacent_inst.append(min_wsnm)
        hos_loc['adj_inst'] = adjacent_inst
        hos_loc.to_csv('./data/hospital_loc_withkey.csv', index=False)


    def merge_tables(self):
        raw = pd.read_csv('./data/no_show.csv')
        key = pd.read_csv('./data/hospital_loc_withkey.csv', usecols=['location','adj_inst'])
        weather = pd.read_csv('./data/sudeste_relevant.csv').fillna(0)

        # preprocess start
        date_lag = []
        weekday = [] # 0 : Monday //// 6 : Sunday
        year = []
        month = []
        season = []  # 0 : spring / 1 : summer / 2 : fall / 3 : winter
        day = []
        age_bin = []  # 0 : 0-10  /  1 : 11-26  /  2 : 27-64  /  3 : 65-
        rain = []
        solar = []
        temp = []
        dewpoint = []
        humidity = []
        for _, row in raw.iterrows():
            print(_)
            schddate = row['ScheduledDay']
            schddate = datetime.datetime(int(schddate[:4]), int(schddate[5:7]), int(schddate[8:10]))
            apmtdate = row['AppointmentDay']
            yr = int(apmtdate[:4])
            mo = int(apmtdate[5:7])
            da = int(apmtdate[8:10])
            apmtdate = datetime.datetime(yr, mo, da)
            date_lag.append((apmtdate-schddate).days)
            weekday.append(apmtdate.weekday())
            year.append(yr)
            month.append(mo)
            day.append(da)

            if row['Age'] <= 10 :
                age_bin.append(0)
            elif row['Age'] <= 26 :
                age_bin.append(1)
            elif row['Age'] <= 64 :
                age_bin.append(2)
            else :
                age_bin.append(3)

            # seasons reference : https://seasonsyear.com/Brazil
            if mo in [3,4,5] :
                season.append(0)
            elif mo in [6,7,8] :
                season.append(1)
            elif mo in [9,10,11] :
                season.append(2)
            elif mo in [12,1,2] :
                season.append(3)

            location = row['Neighbourhood']
            adj_inst = key[key.location == location].adj_inst.item()
            weather_loc = weather[weather.wsnm == adj_inst]
            weather_row = weather_loc[(weather_loc.yr==yr) & (weather_loc.mo==mo) &
                                      (weather_loc.da==da) & (weather_loc.hr==12)]
            if weather_row.shape[0] == 0 :
                rain.append(np.nan)
                solar.append(np.nan)
                temp.append(np.nan)
                dewpoint.append(np.nan)
                humidity.append(np.nan)
            else :
                rain.append(weather_row.prcp.item())
                solar.append(weather_row.gbrd.item())
                temp.append(weather_row.temp.item())
                dewpoint.append(weather_row.dewp.item())
                humidity.append(weather_row.hmdy.item())

        raw['Gender'] = raw['Gender'].map({'M': 1, 'F': 0})
        raw['No-show'] = raw['No-show'].map({'No':0, 'Yes': 1})

        raw['date_lag'] = date_lag
        raw['weekday'] = weekday
        raw['year'] = year
        raw['month'] = month
        raw['season'] = season
        raw['day'] = day
        raw['age_bin'] = age_bin
        raw['rain'] = rain
        raw['solar'] = solar
        raw['temp'] = temp
        raw['dewpoint'] = dewpoint
        raw['humidity'] = humidity


        final = raw[['Gender','Age','age_bin','Neighbourhood','Scholarship','Hipertension','Diabetes','Alcoholism',
                     'Handcap','SMS_received','date_lag','weekday','year','month','day','season','rain','solar','temp',
                     'dewpoint','humidity','No-show']]

        final = final.fillna(final.mean())
        final.to_csv('./data/final.csv',index=False)

    def load_data(self):
        def prepro(data) :
            # get rid of Age 0 and negative date_lag
            data = data[data['Age'] != 0]
            data = data[data['date_lag'] >= 0]
            # normalize date_lag
            dl = np.log(np.array(data['date_lag'])+1)
            data['date_lag'] = dl
            return data

        if self.config.impt_only :
            data = pd.read_csv('./data/final_prepro.csv',
                               usecols=['SMS_received','date_lag','season','Age','age_bin','Scholarship','dewpoint',
                                        'humidity','Alcoholism','temp','Hipertension','weekday','No-show'])
            data_oh = pd.read_csv('./data/final_prepro_onehot.csv',
                                   usecols=['SMS_received','date_lag','season','Age','kid','youth','adult','senior',
                                            'Scholarship','dewpoint','humidity','Alcoholism','temp','Hipertension','mon',
                                            'tue','wed','thu','fri','sat','No-show'])
            data = prepro(data)
            data_oh = prepro(data_oh)
            return data, data_oh
        else :
            data = pd.read_csv('./data/final_prepro.csv')
            data_oh = pd.read_csv('./data/final_prepro_onehot.csv')
            data = prepro(data)
            data_oh = prepro(data_oh)
            return data, data_oh