import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import math

data = {'sHours':['5','5','7','10','5','8','6','8','6','8','7','7','7','7','8','6','3','5','9'],
        'Work':['0','1','1','0','1','0','1','0','0','1','0','0','0','1','0','1','1','0','0'],
        'Weight':['90','82','75','69','100','67','65','88','103','85','62','70','55','65','82','51','82','70','67'],
        'Height':['175','172','177','179','180','170','179','187','184','195','170','175','168','178','190','150','176','177','172'],
        'Way':['2','0.8','1','0.25','2','2','1.5','0.17','1','0.6667','1','1','1','1','1.5','1.5','1.5','1','1.5'],
        'Total':['t','t','c','t','t','t','c','t','t','c','t','c','c','t','c','c','t','c','c']}

frame = pd.DataFrame(data)
frame

sHoursCur = int(input("Сколько часов сна? "))
WorkCur = int(input("Трудоустроен? "))
WeightCur = int(input("Какой вес? "))
HeightCur = int(input("Какой рост? "))
WayCur = float(input("Сколько времени тратите на поездку? "))


for x in data:
  sHours = data['sHours']
  Work = data['Work']
  Weight = data['Weight']
  Height = data['Height']
  Way = data['Way']
  Total = data['Total']

i = 0
dist = []
for x in data['sHours']:
  dist.insert(i, math.sqrt((sHoursCur - int(sHours[i])) ** 2 + (WorkCur - int(Work[i])) ** 2 + (WeightCur - int(Weight[i])) ** 2) + (HeightCur - int(Height[i])) ** 2 + (WayCur - float(Way[i])) ** 2)
  i += 1
  print(dist)

data1 = data
data1['Dist'] =  dist
frame1 = pd.DataFrame(data1)

data2 = data1
frame2 = pd.DataFrame(data2)
frame2 = frame2.sort_values('Dist')
data2 = frame2

i = 0 
numberOfRank = 1
rank = []
for x in data2['sHours']:
  rank.insert(i, numberOfRank)
  i += 1
  numberOfRank += 1
data2["Rank"] = rank

frame3 = pd.DataFrame(data2)


i = 0
TotalTea = 0
TotalCoffee = 0
for x in data2['Rank']:
  TotalEnd = data2['Total']


for x in data2['Rank']:
  if TotalEnd[i] == 't':
    TotalTea += 1
  else:
    TotalCoffee += 1
  i += 1

print(TotalCoffee)
print(TotalTea)
if TotalCoffee > TotalTea:
  print('Пользователь будет предпочитать Кофе')
else:
  print('Пользователь будет предпочитать Чай')
