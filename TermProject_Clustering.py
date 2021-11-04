
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
#model
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture


from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
warnings.filterwarnings(action='ignore')

plt.style.use('ggplot')
df = pd.read_csv('archive (1)/Indicators.csv')
country_df = pd.read_csv('archive (1)/Country.csv')

# Create list of unique indicators, indicator codes
Indicator_array =  df[['IndicatorName','IndicatorCode']].drop_duplicates().values
ind = np.where(Indicator_array[:,1] == 'SM.POP.TOTL.ZS')
print(ind[0].item(0),'The ind')
print(type(Indicator_array))

income_df = country_df.loc[:,'IncomeGroup']
region_df = country_df.loc[:, 'Region']
income_df.index = list(country_df.loc[:,'CountryCode'])
region_df.index = list(country_df.loc[:,'CountryCode'])
income_df.name = "IncomeGroup"
region_df.name = "Region"


class ClusterMaster(object):
    def __init__(self, df=None, scaler=None):
        self.scaler = scaler
        self.encoder = LabelEncoder()
        self.df = df
        self.result = None
    def sef_df(self, df):
        self.df = df

    def set_encoding(self, columns):
        self.df[columns] = self.encoder.fit_transform(df[columns])

    def scaling(self, scaler=None, gt=False):
        if self.scaler is None and scaler is None:
            return 0
        if gt:
            temp = self.scaler.fit_transform(self.df.iloc[:, :-1])
        else:
            temp = self.scaler.fit_transform(self.df)

    def fit_cluster(self, algorithm, param, shil = False, purity=False, plot=False):
        self.model = algorithm(**param)
        X = self.df.iloc[:, :-2]
        Y = self.df.iloc[:, -2:]
        X = self.scaler.fit_transform(X)
        if isinstance(self.model, GaussianMixture):
            result = self.model.fit_predict(X)
        else:
            result = self.model.fit(X)
        if shil:
            if not isinstance(result, np.ndarray):
                result_np = np.reshape(result.labels_, (-1, 1))
            else:
                result_np = np.reshape(result, (-1, 1))
            if(len(np.unique(result_np))==1):
                print("There are just 1 Result...")
            else:
                pca = PCA(n_components=3)
                new_x = pca.fit_transform(X)
                score = silhouette_score(new_x, result_np)
                print(score, '_________________silhouette Score')
        if purity:
            gt_np = np.reshape(Y['Region'].values, (-1, 1))
            gt_np2 = np.reshape(Y['IncomeGroup'].values, (-1, 1))
            if not isinstance(result_np, np.ndarray):
                r = result.labels_
            else:
                r = result_np
            r = np.reshape(r, (-1, 1))
            purity = purity_score(r, gt_np)
            print(purity, 'Region_________________purity Score')
            purity = purity_score(r, gt_np2)
            print(purity, 'IncomeGroup_________________purity Score')
        if plot:
            if not isinstance(result, np.ndarray):
                plot_scatter(X, result.labels_)
            else:
                plot_scatter(X, result_np.squeeze(axis=1))
        if not isinstance(result, np.ndarray):
            result=  result.labels_
        else:
            result = result_np.squeeze(1)
        self.result = pd.Series(result, name="Result", index=self.df.index)
        return result

    def save_df(self, path):
        print('Save_df')
        self.df.to_csv(path, mode='a', header=True)

    def statistic_per_cluster(self, idc_translator, gt=True):
        if self.result is None:
            return 0
        if gt:
            x = self.df.iloc[:, :-1]
        else:
            x = self.df.copy()

        result_df = pd.concat([x,self.result], axis=1)
        print(result_df['Result'].unique())
        result_group = result_df.groupby(result_df['Result'], axis=0)
        print(result_group.mean(),'Group Means')
    def getPercentage_with_col(self, df, col):
        temp_df = pd.concat([df, self.result], axis=1)
        temp_df[col] = label_object[col].inverse_transform(df[col])
        group_list = temp_df.groupby(temp_df['Result'], axis=0)
        for idx, group in enumerate(group_list):
            group = group[1]
            total_num = len(group)
            unique_list = group[col].unique()
            print("Cluster...%d"%(idx+1))
            for u in unique_list:
                sub_num = len(group[group[col]==u])
                percentage = sub_num/total_num*100
                print("%s has %f percentage"%(u, percentage))


def weighted_mean(array):
    def inner(x):
        return (array*x).mean()
    return inner;


# Part that determines how to handle na value of dataframe
def handleNa(col, df,work="mean"):
    # fill na value for 'mean'
    if work=="mean":
        for c in col:
            mean = df[c].mean()
            df[c]=df[c].fillna(mean)
    # fill na value for 'median'
    elif work=="median":
        for c in col:
            median = df[c].median()
            df[c]=df[c].fillna(median)
    # fill na value for 'mode'
    elif work=="mode":
        for c in col:
            mode = df[c].mode()[0]
            df[c]=df[c].fillna(mode)
    # drop row which contains na value
    elif work=="drop":
        df = df.dropna(subset=col)
    return df

def purity_score(predict, gt):
    contingency_m = contingency_matrix(gt, predict)
    return np.sum(np.amax(contingency_m, axis=0))/ np.sum(contingency_m)

def plot_scatter(df, predict, dim=3):
    fig = pyplot.figure()
    ax = Axes3D(fig)

    pca = PCA(n_components=dim)
    dep_df = pca.fit_transform(df)
    ax.scatter(dep_df[:, 0], dep_df[:, 1], dep_df[:, 2], c=predict)
    pyplot.show()

def process(df, scaler, algorithm):
    X = df.iloc[:, :-2]
    Y = df.iloc[:, -2:]#Region, IncomeGroup
    X = scaler.fit_transform(X)
    result = algorithm.fit(X)
    result_np = np.reshape(result.labels_, (-1, 1))
    #Y = np.reshape(Y, (-1, 1))
    Y = np.reshape(Y.values, (-1, 1))
    #score = silhouette_score(X, result_np)
    purity = purity_score(result.labels_, Y)
    #print(score, '_________________silhouette Score')
    print(purity, '_________________purity Score')
    plot_scatter(X, result.labels_)
    return result.labels_

def process_mixture(df, scaler, algorithm):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X = scaler.fit_transform(X)
    print(X, 'itis x itis x itis x itis x')
    result = algorithm.fit_predict(X)
    result_np = np.reshape(result, (-1, 1))
    # Y = np.reshape(Y, (-1, 1))
    Y = np.reshape(Y.values, (-1, 1))
    # score = silhouette_score(X, result_np)
    purity = purity_score(result, Y)
    # print(score, '_________________silhouette Score')
    print(purity, '_________________purity Score')
    plot_scatter(X, result)

modified_indicators = []
unique_indicator_codes = []
for ele in Indicator_array:
    indicator = ele[0]
    indicator_code = ele[1].strip()
    if indicator_code not in unique_indicator_codes:
        # delete , ( ) from the IndicatorNames
        new_indicator = re.sub('[,()]',"",indicator).lower()
        # replace - with "to" and make all words into lower case
        new_indicator = re.sub('-'," to ",new_indicator).lower()
        modified_indicators.append([new_indicator,indicator_code])
        unique_indicator_codes.append(indicator_code)

Indicators = pd.DataFrame(modified_indicators,columns=['IndicatorName','IndicatorCode'])
Indicators = Indicators.drop_duplicates()
print(Indicators.shape)

key_word_dict = {}
key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']
key_word_dict['Economy'] = ['income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']
key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
key_word_dict['Education'] = ['education','literacy']
key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']


key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']


key_word_dict['Food'] = ['food','grain','nutrition','calories']#
key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']#
key_word_dict['Rural'] = ['rural','village']#
key_word_dict['Urban'] = ['urban','city']# 4개 이용.


feature = ['Demography', 'Health', 'Economy']

total_list = []
for indicator_ele in Indicators.values:#처음에 df에는 코드, 이름만 있어~~ 우리는 우리가 쓸거 따로 찾아야함.
    for f in feature:
        for ele in key_word_dict[f]:
            word_list = indicator_ele[0].split()
            if ele in word_list or ele+'s' in word_list:
                # Uncomment this line to print the indicators explicitely
                total_list.append(indicator_ele[1])
                break
print(total_list)
print(len(total_list))
df_subset = df[df['IndicatorCode'].isin(total_list)]

country_list = df_subset['CountryCode'].unique()
year_list = df_subset['Year'].unique()
print('Year Min is... %d', year_list.min())
year_min = year_list.min()-1
df_subset.loc[:,'Year']=df_subset['Year']-year_min
print(df_subset['Year'])

print("There are %d Countries!"%(len(country_list)))
print("There are %d Years!"%(len(year_list)))
atomic_df = df_subset[['CountryCode','Year',  'IndicatorCode', 'Value']]
#split by country
split_df=[]
weighted_feature = df_subset['Year'].copy() * df_subset['Value'].copy()
atomic_df.loc[:, 'WValue'] = weighted_feature

for country in country_list:
    s_df = atomic_df[atomic_df['CountryCode']==country]
    means = s_df.groupby('IndicatorCode')['WValue'].mean()
    key = list(means.keys())
    value = means.values[:]
    s_df = pd.DataFrame([value], columns=key, index=[country])
    split_df.append(s_df)

#concat에 따라 target이 다르다.
total_list = pd.concat(split_df)
income_df = income_df.fillna("Unknown")
region_df = region_df.fillna("Unknown")
total_list = pd.concat([total_list, income_df], axis=1)
total_list = pd.concat([total_list, region_df], axis=1)

total_list_valid = total_list.copy()
print("fill_na is Done")
#total_list_valid = total_list_valid.dropna(subset=['IncomeGroup'])
#total_list_predict = total_list[total_list['IncomeGroup']==np.nan]

print(total_list_valid, 'for training data')

#
threshold = len(total_list_valid.iloc[:,0])//4
print(threshold, '<--- Threshold')
for col in total_list_valid:
    if total_list_valid[col].isna().sum().sum()> threshold:
        print(col, 'is dropped')
        total_list_valid = total_list_valid.drop([col], axis=1)
print(total_list_valid.shape, 'Shape of Column')
print(total_list_valid.isna().sum(), 'Current Features')

l_encoder = LabelEncoder()
l_encoder2 = LabelEncoder()
print(total_list_valid.columns)
label_object = {}
total_list_valid['Region'] = l_encoder.fit_transform(total_list_valid['Region'])
label_object['Region']= l_encoder
total_list_valid['IncomeGroup'] = l_encoder2.fit_transform(total_list_valid['IncomeGroup'])
label_object['IncomeGroup'] = l_encoder2
total_list_valid = handleNa(list(total_list_valid.columns), total_list_valid)


#Scaling Process
s_scaler = StandardScaler()
m_scaler = MinMaxScaler()

kmeans = KMeans#income group
dbscan = DBSCAN# default_ 하나로 예측, 좋은결과 얻지 못함
em = GaussianMixture

cluster_model = ClusterMaster(total_list_valid, scaler=m_scaler)
cluster_model.scaling(gt=True)

cluster_model.fit_cluster(algorithm=kmeans, param={'n_clusters':5}, shil=True, purity=True, plot=True)
cluster_model.getPercentage_with_col(total_list_valid, 'IncomeGroup')
cluster_model.getPercentage_with_col(total_list_valid, 'Region')

#cluster_model.save_df('kmeans.csv')
#cluster_model.statistic_per_cluster(Indicator_array)

cluster_model.statistic_per_cluster(Indicator_array)
cluster_model.fit_cluster(algorithm=dbscan, param={'eps':1.36, 'min_samples':5}, shil=True, purity=True, plot=True)
cluster_model.getPercentage_with_col(total_list_valid, 'IncomeGroup')
cluster_model.getPercentage_with_col(total_list_valid, 'Region')
print("=================================================")
cluster_model.statistic_per_cluster(Indicator_array)
cluster_model.fit_cluster(algorithm=dbscan, param={'eps':1.36, 'min_samples':4}, shil=True, purity=True, plot=True)
cluster_model.getPercentage_with_col(total_list_valid, 'IncomeGroup')
cluster_model.getPercentage_with_col(total_list_valid, 'Region')


#cluster_model.save_df('dbscan.csv')
cluster_model.fit_cluster(algorithm=em, param={'n_components':5}, shil=True, purity=True, plot=True)
cluster_model.getPercentage_with_col(total_list_valid, 'IncomeGroup')
cluster_model.getPercentage_with_col(total_list_valid, 'Region')
#cluster_model.save_df('em.csv')

def set_param(name, param):
    dic = {}
    for idx, n in enumerate(name):
        dic[n] = param[idx]
    return dic
'''
kmeans = KMeans(n_clusters=5)#income group
dbscan = DBSCAN()# default_ 하나로 예측, 좋은결과 얻지 못함
em = GaussianMixture(n_components=5)
process(total_list_valid, m_scaler, dbscan)
#process_mixture(total_list_valid, m_scaler, em)
'''