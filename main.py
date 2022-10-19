import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
import openpyxl
filterwarnings("ignore")

#import sns as sns

filterwarnings("ignore")

df = pd.read_excel('Gruppy_22__1,xlsx/Gruppy_22__1.xlsx')
df.head()

df.info()

# посмотрим на визуализацию
sns.pairplot(data = df,hue = 'К/Ч')

# преобразуем бинарные значения
dict_decode = {0:'чай',1:'кофе'}
df['К/Ч'] = df['К/Ч'].replace({'ч':0,'к':1})
df['мама'] = df.iloc[:,5].map(lambda x:x[0]).replace({'ч':0,'к':1})
df['папа'] = df.iloc[:,5].map(lambda x:x[1]).replace({'ч':0,'к':1})
df = df.drop(columns=['Родители (мама, папа) что пьют'])
df.head()


# оставим пару примеров для теста
train_idx = np.random.choice(np.arange(19),size = 17,replace=False)
test_idx = list(set(np.arange(19))  - set(train_idx))
train = df.iloc[train_idx]
test = df.iloc[test_idx]


def determine_preference_of_drink_by_knn(train_df, new_object, k=5, type_norm=2):
    help_dict = {0: 'Чай', 1: 'Кофе'}

    train_df['distance_to_new_object'] = (np.linalg.
                                          norm(train_df.drop(columns=['К/Ч']) - new_object,
                                               ord=type_norm, axis=1))

    answer = train_df.sort_values('distance_to_new_object').iloc[:k]['К/Ч'].value_counts().index.tolist()[0]
    #     .mode().values[0]  можно и так
    return f'Человек с таким набором характеристик предпочитает пить {help_dict[answer]}'

print(determine_preference_of_drink_by_knn(train,test.iloc[0,[0,1,2,3,4,6,7]]))
print('Указанное предпочтение',dict_decode[test.iloc[0,5]])

time_sleep = int(input('Часы сна:'))
work = int(input('Трудоустройство:'))
weight = int(input('Вес:'))
height = int(input('Рост:'))

distance = float(input('Расстояние до МИРЭА в часах:'))
mama = input('Что пьет мама:')
papa = input('Что пьет папа:')
dict_encode = {'чай':0,'кофе':1}
test_data = np.array([time_sleep,work,weight,height,distance,
                      dict_encode[mama.lower()],dict_encode[papa.lower()]])
print(determine_preference_of_drink_by_knn(df,test_data))
print('Самое близкое расстояние',sorted(df.distance_to_new_object)[0])
print(df.sort_values(by='distance_to_new_object'))
