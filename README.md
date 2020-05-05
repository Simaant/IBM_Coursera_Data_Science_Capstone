
# IBM Data Science Capstone Project

**The project focuses on the neighbourhoods and boroughs in New York City and aims to solve the problem of selecting an appropriate neighbourhood for opening a Chinese restaurant.**

I've utilized word clouds on the cuisine data of NYC obtained from wikipedia to understand the spread of cuisines. I utilized the results of the word cloud to select a location in NYC and implemented K-means clustering in that region.  


```python
#Importing the required libraries

import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analsysis
from PIL import Image # converting images into arrays

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes 
#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')
%matplotlib inline

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
```

    Libraries imported.
    Matplotlib version:  3.0.2
    


```python
#Importing the word cloud library

from wordcloud import WordCloud, STOPWORDS
```

## Exploring the cuisine dataset


```python
cuisine=pd.read_csv('C:\\Users\\simaa\\Desktop\\Projects\\IBM_Coursera_Data_Science_Capstone\\data\\Restaurant_Grades.csv')
```


```python
cuisine.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DBA</th>
      <th>BORO</th>
      <th>BUILDING</th>
      <th>STREET</th>
      <th>ZIPCODE</th>
      <th>CUISINE DESCRIPTION</th>
      <th>VIOLATION DESCRIPTION</th>
      <th>CRITICAL FLAG</th>
      <th>SCORE</th>
      <th>GRADE</th>
      <th>GRADE DATE</th>
      <th>RECORD DATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MORRIS PARK BAKE SHOP</td>
      <td>Bronx</td>
      <td>1007</td>
      <td>MORRIS PARK AVE</td>
      <td>10462.0</td>
      <td>Bakery</td>
      <td>Pesticide use not in accordance with label or ...</td>
      <td>N</td>
      <td>6</td>
      <td>A</td>
      <td>06/11/2019</td>
      <td>08/13/2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MORRIS PARK BAKE SHOP</td>
      <td>Bronx</td>
      <td>1007</td>
      <td>MORRIS PARK AVE</td>
      <td>10462.0</td>
      <td>Bakery</td>
      <td>Plumbing not properly installed or maintained;...</td>
      <td>N</td>
      <td>6</td>
      <td>A</td>
      <td>06/11/2019</td>
      <td>08/13/2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MORRIS PARK BAKE SHOP</td>
      <td>Bronx</td>
      <td>1007</td>
      <td>MORRIS PARK AVE</td>
      <td>10462.0</td>
      <td>Bakery</td>
      <td>Non-food contact surface improperly constructe...</td>
      <td>N</td>
      <td>6</td>
      <td>A</td>
      <td>06/11/2019</td>
      <td>08/13/2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MORRIS PARK BAKE SHOP</td>
      <td>Bronx</td>
      <td>1007</td>
      <td>MORRIS PARK AVE</td>
      <td>10462.0</td>
      <td>Bakery</td>
      <td>Pesticide use not in accordance with label or ...</td>
      <td>N</td>
      <td>5</td>
      <td>A</td>
      <td>05/11/2018</td>
      <td>08/13/2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MORRIS PARK BAKE SHOP</td>
      <td>Bronx</td>
      <td>1007</td>
      <td>MORRIS PARK AVE</td>
      <td>10462.0</td>
      <td>Bakery</td>
      <td>Non-food contact surface improperly constructe...</td>
      <td>N</td>
      <td>5</td>
      <td>A</td>
      <td>05/11/2018</td>
      <td>08/13/2019</td>
    </tr>
  </tbody>
</table>
</div>




```python
bor_cuisine=cuisine[['BORO', 'CUISINE DESCRIPTION']]
```


```python
type(bor_cuisine)
```




    pandas.core.frame.DataFrame




```python
bor_cuisine.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BORO</th>
      <th>CUISINE DESCRIPTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>Bakery</td>
    </tr>
  </tbody>
</table>
</div>




```python
bor_cuisine['BORO'].value_counts()
```




    Manhattan        76389
    Brooklyn         48689
    Queens           44702
    Bronx            18154
    Staten Island     6675
    0                   54
    Name: BORO, dtype: int64




```python
bor_cuisine['CUISINE DESCRIPTION'].value_counts()
```




    American                                                            43124
    Chinese                                                             19892
    Café/Coffee/Tea                                                     10510
    Pizza                                                                8712
    Italian                                                              7787
    Latin (Cuban, Dominican, Puerto Rican, South & Central American)     7687
    Mexican                                                              7540
    Japanese                                                             6637
    Bakery                                                               6199
    Caribbean                                                            6180
    Spanish                                                              5493
    Pizza/Italian                                                        4081
    Chicken                                                              3704
    Donuts                                                               3471
    Delicatessen                                                         2981
    Indian                                                               2957
    Asian                                                                2942
    Hamburgers                                                           2746
    Jewish/Kosher                                                        2721
    Thai                                                                 2440
    Sandwiches                                                           2434
    Juice, Smoothies, Fruit Salads                                       2419
    Korean                                                               2391
    French                                                               2298
    Mediterranean                                                        2033
    Ice Cream, Gelato, Yogurt, Ices                                      1811
    Irish                                                                1738
    Sandwiches/Salads/Mixed Buffet                                       1612
    Middle Eastern                                                       1419
    Seafood                                                              1417
    Bagels/Pretzels                                                      1370
    Greek                                                                1122
    Tex-Mex                                                               999
    Vegetarian                                                            888
    Vietnamese/Cambodian/Malaysia                                         853
    Peruvian                                                              815
    Other                                                                 743
    African                                                               689
    Steak                                                                 640
    Bottled beverages, including water, sodas, juices, etc.               616
    Eastern European                                                      577
    Salads                                                                546
    Russian                                                               541
    Turkish                                                               540
    Soul Food                                                             495
    Bangladeshi                                                           494
    Chinese/Japanese                                                      428
    Continental                                                           374
    Barbecue                                                              370
    Soups & Sandwiches                                                    300
    Pakistani                                                             293
    Filipino                                                              279
    Polish                                                                240
    Tapas                                                                 234
    Brazilian                                                             229
    Hawaiian                                                              224
    Creole                                                                217
    German                                                                216
    Armenian                                                              178
    Australian                                                            172
    Chinese/Cuban                                                         171
    Hotdogs/Pretzels                                                      146
    Hotdogs                                                               127
    Ethiopian                                                             124
    Pancakes/Waffles                                                      124
    English                                                               109
    Afghan                                                                 98
    Egyptian                                                               81
    Portuguese                                                             79
    Moroccan                                                               69
    Not Listed/Not Applicable                                              67
    Creole/Cajun                                                           47
    Indonesian                                                             47
    Californian                                                            44
    Cajun                                                                  44
    Scandinavian                                                           43
    Fruits/Vegetables                                                      39
    Soups                                                                  33
    Iranian                                                                29
    Southwestern                                                           26
    Nuts/Confectionary                                                     21
    Czech                                                                  20
    Chilean                                                                13
    Basque                                                                  4
    Name: CUISINE DESCRIPTION, dtype: int64



#### Exploring cuisines in Manhattan


```python
Man_cuisine=bor_cuisine[bor_cuisine['BORO'] == 'Manhattan']
Cuisine_M=Man_cuisine[['CUISINE DESCRIPTION']]
Cuisine_M.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUISINE DESCRIPTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Irish</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Irish</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Irish</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Irish</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Irish</td>
    </tr>
  </tbody>
</table>
</div>




```python
Cuisine_M.to_csv('CUISINE_Manhattan.txt', sep=',', index=False)
```


```python
Cuisine_M1 = open('C:\\Users\\simaa\\Desktop\\Projects\\IBM_Coursera_Data_Science_Capstone\\results\\CUISINE_Manhattan.txt', 'r').read()
```


```python
stopwords = set(STOPWORDS)
```


```python
NYC_CUISINE_Man = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
NYC_CUISINE_Man.generate(Cuisine_M1)
```




    <wordcloud.wordcloud.WordCloud at 0x28934776128>




```python
plt.imshow(NYC_CUISINE_Man, interpolation='bilinear')
plt.axis('off')

fig = plt.figure()
fig.set_figwidth(45)
fig.set_figheight(55)

plt.show()
```


![png](visualizations/Manhattan_WC.PNG)



    <Figure size 3240x3960 with 0 Axes>


#### Exploring cuisines in Brooklyn 


```python
Brook_cuisine=bor_cuisine[bor_cuisine['BORO'] == 'Brooklyn']
Cuisine_B=Brook_cuisine[['CUISINE DESCRIPTION']]
Cuisine_B.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUISINE DESCRIPTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Hamburgers</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hamburgers</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Hamburgers</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Hamburgers</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hamburgers</td>
    </tr>
  </tbody>
</table>
</div>




```python
Cuisine_B.to_csv('CUISINE_Brooklyn.txt', sep=',', index=False)
```


```python
Cuisine_B1 = open('C:\\Users\\simaa\\Desktop\\Projects\\IBM_Coursera_Data_Science_Capstone\\results\\CUISINE_Brooklyn.txt', 'r').read()
```


```python
NYC_CUISINE_B = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
NYC_CUISINE_B.generate(Cuisine_B1)
```




    <wordcloud.wordcloud.WordCloud at 0x28934788f28>




```python
plt.imshow(NYC_CUISINE_B, interpolation='bilinear')
plt.axis('off')

fig = plt.figure()
fig.set_figwidth(45)
fig.set_figheight(55)

plt.show()
```


![png](visualizations/WC_Brok.png)



    <Figure size 3240x3960 with 0 Axes>


#### Exploring cuisines in Bronx


```python
Bronx_cuisine=bor_cuisine[bor_cuisine['BORO'] == 'Bronx']
Cuisine_Br=Bronx_cuisine[['CUISINE DESCRIPTION']]
Cuisine_Br.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUISINE DESCRIPTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bakery</td>
    </tr>
  </tbody>
</table>
</div>




```python
Cuisine_Br.to_csv('CUISINE_Bronx.txt', sep=',', index=False)
```


```python
Cuisine_Br1 = open('C:\\Users\\simaa\\Desktop\\Projects\\IBM_Coursera_Data_Science_Capstone\\results\\CUISINE_Bronx.txt', 'r').read()

NYC_CUISINE_Br = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
NYC_CUISINE_Br.generate(Cuisine_Br1)
```




    <wordcloud.wordcloud.WordCloud at 0x28935e0e278>




```python
plt.imshow(NYC_CUISINE_Br, interpolation='bilinear')
plt.axis('off')

fig = plt.figure()
fig.set_figwidth(45)
fig.set_figheight(55)

plt.show()
```


![png](visualizations/WC_Bronx.png)



    <Figure size 3240x3960 with 0 Axes>


#### Exploring cuisines on Staten Island


```python
staten_island_cuisine=bor_cuisine[bor_cuisine['BORO'] == 'Staten Island']
Cuisine_stat=staten_island_cuisine[['CUISINE DESCRIPTION']]
Cuisine_stat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUISINE DESCRIPTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>201</th>
      <td>Delicatessen</td>
    </tr>
    <tr>
      <th>202</th>
      <td>Delicatessen</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Delicatessen</td>
    </tr>
    <tr>
      <th>204</th>
      <td>Delicatessen</td>
    </tr>
    <tr>
      <th>205</th>
      <td>Delicatessen</td>
    </tr>
  </tbody>
</table>
</div>




```python
Cuisine_stat.to_csv('CUISINE_StatenIsland.txt', sep=',', index=False)
```


```python
Cuisine_stat1 = open('C:\\Users\\simaa\\Desktop\\Projects\\IBM_Coursera_Data_Science_Capstone\\results\\CUISINE_StatenIsland.txt', 'r').read()

NYC_CUISINE_stat = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
NYC_CUISINE_stat.generate(Cuisine_stat1)
```




    <wordcloud.wordcloud.WordCloud at 0x28935807be0>




```python
plt.imshow(NYC_CUISINE_stat, interpolation='bilinear')
plt.axis('off')

fig = plt.figure()
fig.set_figwidth(45)
fig.set_figheight(55)

plt.show()
```


![png](visualizations/WC_Stat_Is.png)



    <Figure size 3240x3960 with 0 Axes>


**The above word clouds show that the most popular cuisine in New York in American. In case of Chinese cuisine, Bronx has more chinese restaurants than any other city in the NYC region. Thus, exploring Bronx makes sense.**



```python
with open('C:\\Users\\simaa\\Desktop\\Projects\\IBM_Coursera_Data_Science_Capstone\\data\\newyork_data.json') as json_data:
    newyork_data = json.load(json_data)
```


```python
newyork_data
```




    {'type': 'FeatureCollection',
     'totalFeatures': 306,
     'features': [{'type': 'Feature',
       'id': 'nyu_2451_34572.1',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84720052054902, 40.89470517661]},
       'geometry_name': 'geom',
       'properties': {'name': 'Wakefield',
        'stacked': 1,
        'annoline1': 'Wakefield',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.84720052054902,
         40.89470517661,
         -73.84720052054902,
         40.89470517661]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.2',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82993910812398, 40.87429419303012]},
       'geometry_name': 'geom',
       'properties': {'name': 'Co-op City',
        'stacked': 2,
        'annoline1': 'Co-op',
        'annoline2': 'City',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.82993910812398,
         40.87429419303012,
         -73.82993910812398,
         40.87429419303012]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.3',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82780644716412, 40.887555677350775]},
       'geometry_name': 'geom',
       'properties': {'name': 'Eastchester',
        'stacked': 1,
        'annoline1': 'Eastchester',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.82780644716412,
         40.887555677350775,
         -73.82780644716412,
         40.887555677350775]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.4',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90564259591682, 40.89543742690383]},
       'geometry_name': 'geom',
       'properties': {'name': 'Fieldston',
        'stacked': 1,
        'annoline1': 'Fieldston',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90564259591682,
         40.89543742690383,
         -73.90564259591682,
         40.89543742690383]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.5',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9125854610857, 40.890834493891305]},
       'geometry_name': 'geom',
       'properties': {'name': 'Riverdale',
        'stacked': 1,
        'annoline1': 'Riverdale',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.9125854610857,
         40.890834493891305,
         -73.9125854610857,
         40.890834493891305]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.6',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90281798724604, 40.88168737120521]},
       'geometry_name': 'geom',
       'properties': {'name': 'Kingsbridge',
        'stacked': 1,
        'annoline1': 'Kingsbridge',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90281798724604,
         40.88168737120521,
         -73.90281798724604,
         40.88168737120521]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.7',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91065965862981, 40.87655077879964]},
       'geometry_name': 'geom',
       'properties': {'name': 'Marble Hill',
        'stacked': 2,
        'annoline1': 'Marble',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.91065965862981,
         40.87655077879964,
         -73.91065965862981,
         40.87655077879964]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.8',
       'geometry': {'type': 'Point',
        'coordinates': [-73.86731496814176, 40.89827261213805]},
       'geometry_name': 'geom',
       'properties': {'name': 'Woodlawn',
        'stacked': 1,
        'annoline1': 'Woodlawn',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.86731496814176,
         40.89827261213805,
         -73.86731496814176,
         40.89827261213805]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.9',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8793907395681, 40.87722415599446]},
       'geometry_name': 'geom',
       'properties': {'name': 'Norwood',
        'stacked': 1,
        'annoline1': 'Norwood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.8793907395681,
         40.87722415599446,
         -73.8793907395681,
         40.87722415599446]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.10',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85744642974207, 40.88103887819211]},
       'geometry_name': 'geom',
       'properties': {'name': 'Williamsbridge',
        'stacked': 1,
        'annoline1': 'Williamsbridge',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.85744642974207,
         40.88103887819211,
         -73.85744642974207,
         40.88103887819211]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.11',
       'geometry': {'type': 'Point',
        'coordinates': [-73.83579759808117, 40.866858107252696]},
       'geometry_name': 'geom',
       'properties': {'name': 'Baychester',
        'stacked': 1,
        'annoline1': 'Baychester',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.83579759808117,
         40.866858107252696,
         -73.83579759808117,
         40.866858107252696]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.12',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85475564017999, 40.85741349808865]},
       'geometry_name': 'geom',
       'properties': {'name': 'Pelham Parkway',
        'stacked': 1,
        'annoline1': 'Pelham Parkway',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.85475564017999,
         40.85741349808865,
         -73.85475564017999,
         40.85741349808865]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.13',
       'geometry': {'type': 'Point',
        'coordinates': [-73.78648845267413, 40.84724670491813]},
       'geometry_name': 'geom',
       'properties': {'name': 'City Island',
        'stacked': 2,
        'annoline1': 'City',
        'annoline2': 'Island',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.78648845267413,
         40.84724670491813,
         -73.78648845267413,
         40.84724670491813]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.14',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8855121841913, 40.870185164975325]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bedford Park',
        'stacked': 2,
        'annoline1': 'Bedford',
        'annoline2': 'Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.8855121841913,
         40.870185164975325,
         -73.8855121841913,
         40.870185164975325]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.15',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9104159619131, 40.85572707719664]},
       'geometry_name': 'geom',
       'properties': {'name': 'University Heights',
        'stacked': 2,
        'annoline1': 'University',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.9104159619131,
         40.85572707719664,
         -73.9104159619131,
         40.85572707719664]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.16',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91967159119565, 40.84789792606271]},
       'geometry_name': 'geom',
       'properties': {'name': 'Morris Heights',
        'stacked': 2,
        'annoline1': 'Morris',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.91967159119565,
         40.84789792606271,
         -73.91967159119565,
         40.84789792606271]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.17',
       'geometry': {'type': 'Point',
        'coordinates': [-73.89642655981623, 40.86099679638654]},
       'geometry_name': 'geom',
       'properties': {'name': 'Fordham',
        'stacked': 1,
        'annoline1': 'Fordham',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.89642655981623,
         40.86099679638654,
         -73.89642655981623,
         40.86099679638654]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.18',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88735617532338, 40.84269615786053]},
       'geometry_name': 'geom',
       'properties': {'name': 'East Tremont',
        'stacked': 2,
        'annoline1': 'East',
        'annoline2': 'Tremont',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.88735617532338,
         40.84269615786053,
         -73.88735617532338,
         40.84269615786053]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.19',
       'geometry': {'type': 'Point',
        'coordinates': [-73.87774474910545, 40.83947505672653]},
       'geometry_name': 'geom',
       'properties': {'name': 'West Farms',
        'stacked': 2,
        'annoline1': 'West',
        'annoline2': 'Farms',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.87774474910545,
         40.83947505672653,
         -73.87774474910545,
         40.83947505672653]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.20',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9261020935813, 40.836623010706056]},
       'geometry_name': 'geom',
       'properties': {'name': 'High  Bridge',
        'stacked': 1,
        'annoline1': 'Highbridge',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.9261020935813,
         40.836623010706056,
         -73.9261020935813,
         40.836623010706056]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.21',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90942160757436, 40.819754370594936]},
       'geometry_name': 'geom',
       'properties': {'name': 'Melrose',
        'stacked': 1,
        'annoline1': 'Melrose',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90942160757436,
         40.819754370594936,
         -73.90942160757436,
         40.819754370594936]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.22',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91609987487575, 40.80623874935177]},
       'geometry_name': 'geom',
       'properties': {'name': 'Mott Haven',
        'stacked': 1,
        'annoline1': 'Mott Haven',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.91609987487575,
         40.80623874935177,
         -73.91609987487575,
         40.80623874935177]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.23',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91322139386135, 40.801663627756206]},
       'geometry_name': 'geom',
       'properties': {'name': 'Port Morris',
        'stacked': 2,
        'annoline1': 'Port',
        'annoline2': 'Morris',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.91322139386135,
         40.801663627756206,
         -73.91322139386135,
         40.801663627756206]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.24',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8957882009446, 40.81509904545822]},
       'geometry_name': 'geom',
       'properties': {'name': 'Longwood',
        'stacked': 1,
        'annoline1': 'Longwood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.8957882009446,
         40.81509904545822,
         -73.8957882009446,
         40.81509904545822]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.25',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88331505955291, 40.80972987938709]},
       'geometry_name': 'geom',
       'properties': {'name': 'Hunts Point',
        'stacked': 2,
        'annoline1': 'Hunts',
        'annoline2': 'Point',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.88331505955291,
         40.80972987938709,
         -73.88331505955291,
         40.80972987938709]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.26',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90150648943059, 40.82359198585534]},
       'geometry_name': 'geom',
       'properties': {'name': 'Morrisania',
        'stacked': 1,
        'annoline1': 'Morrisania',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90150648943059,
         40.82359198585534,
         -73.90150648943059,
         40.82359198585534]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.27',
       'geometry': {'type': 'Point',
        'coordinates': [-73.86574609554924, 40.821012197914015]},
       'geometry_name': 'geom',
       'properties': {'name': 'Soundview',
        'stacked': 1,
        'annoline1': 'Soundview',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.86574609554924,
         40.821012197914015,
         -73.86574609554924,
         40.821012197914015]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.28',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85414416189266, 40.80655112003589]},
       'geometry_name': 'geom',
       'properties': {'name': 'Clason Point',
        'stacked': 2,
        'annoline1': 'Clason',
        'annoline2': 'Point',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.85414416189266,
         40.80655112003589,
         -73.85414416189266,
         40.80655112003589]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.29',
       'geometry': {'type': 'Point',
        'coordinates': [-73.81635002158441, 40.81510925804005]},
       'geometry_name': 'geom',
       'properties': {'name': 'Throgs Neck',
        'stacked': 1,
        'annoline1': 'Throgs Neck',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.81635002158441,
         40.81510925804005,
         -73.81635002158441,
         40.81510925804005]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.30',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8240992675385, 40.844245936947374]},
       'geometry_name': 'geom',
       'properties': {'name': 'Country Club',
        'stacked': 2,
        'annoline1': 'Country',
        'annoline2': 'Club',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.8240992675385,
         40.844245936947374,
         -73.8240992675385,
         40.844245936947374]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.31',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85600310535783, 40.837937822267286]},
       'geometry_name': 'geom',
       'properties': {'name': 'Parkchester',
        'stacked': 1,
        'annoline1': 'Parkchester',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.85600310535783,
         40.837937822267286,
         -73.85600310535783,
         40.837937822267286]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.32',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84219407604444, 40.8406194964327]},
       'geometry_name': 'geom',
       'properties': {'name': 'Westchester Square',
        'stacked': 2,
        'annoline1': 'Westchester',
        'annoline2': 'Square',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.84219407604444,
         40.8406194964327,
         -73.84219407604444,
         40.8406194964327]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.33',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8662991807561, 40.84360847124718]},
       'geometry_name': 'geom',
       'properties': {'name': 'Van Nest',
        'stacked': 2,
        'annoline1': 'Van',
        'annoline2': 'Nest',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.8662991807561,
         40.84360847124718,
         -73.8662991807561,
         40.84360847124718]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.34',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85040178030421, 40.847549063536334]},
       'geometry_name': 'geom',
       'properties': {'name': 'Morris Park',
        'stacked': 1,
        'annoline1': 'Morris Park',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.85040178030421,
         40.847549063536334,
         -73.85040178030421,
         40.847549063536334]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.35',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88845196134804, 40.85727710073895]},
       'geometry_name': 'geom',
       'properties': {'name': 'Belmont',
        'stacked': 1,
        'annoline1': 'Belmont',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.88845196134804,
         40.85727710073895,
         -73.88845196134804,
         40.85727710073895]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.36',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91719048210393, 40.88139497727086]},
       'geometry_name': 'geom',
       'properties': {'name': 'Spuyten Duyvil',
        'stacked': 2,
        'annoline1': 'Spuyten',
        'annoline2': 'Duyvil',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.91719048210393,
         40.88139497727086,
         -73.91719048210393,
         40.88139497727086]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.37',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90453054908927, 40.90854282950666]},
       'geometry_name': 'geom',
       'properties': {'name': 'North Riverdale',
        'stacked': 2,
        'annoline1': 'North',
        'annoline2': 'Riverdale',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90453054908927,
         40.90854282950666,
         -73.90453054908927,
         40.90854282950666]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.38',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8320737824047, 40.85064140940335]},
       'geometry_name': 'geom',
       'properties': {'name': 'Pelham Bay',
        'stacked': 2,
        'annoline1': 'Pelham',
        'annoline2': 'Bay',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.8320737824047,
         40.85064140940335,
         -73.8320737824047,
         40.85064140940335]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.39',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82620275994073, 40.82657951686922]},
       'geometry_name': 'geom',
       'properties': {'name': 'Schuylerville',
        'stacked': 1,
        'annoline1': 'Schuylerville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.82620275994073,
         40.82657951686922,
         -73.82620275994073,
         40.82657951686922]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.40',
       'geometry': {'type': 'Point',
        'coordinates': [-73.81388514428619, 40.821986118163494]},
       'geometry_name': 'geom',
       'properties': {'name': 'Edgewater Park',
        'stacked': 2,
        'annoline1': 'Edgewater',
        'annoline2': 'Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.81388514428619,
         40.821986118163494,
         -73.81388514428619,
         40.821986118163494]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.41',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84802729582735, 40.819014376988314]},
       'geometry_name': 'geom',
       'properties': {'name': 'Castle Hill',
        'stacked': 2,
        'annoline1': 'Castle',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.84802729582735,
         40.819014376988314,
         -73.84802729582735,
         40.819014376988314]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.42',
       'geometry': {'type': 'Point',
        'coordinates': [-73.86332361652777, 40.87137078192371]},
       'geometry_name': 'geom',
       'properties': {'name': 'Olinville',
        'stacked': 1,
        'annoline1': 'Olinville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.86332361652777,
         40.87137078192371,
         -73.86332361652777,
         40.87137078192371]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.43',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84161194831223, 40.86296562477998]},
       'geometry_name': 'geom',
       'properties': {'name': 'Pelham Gardens',
        'stacked': 2,
        'annoline1': 'Pelham',
        'annoline2': 'Gardens',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.84161194831223,
         40.86296562477998,
         -73.84161194831223,
         40.86296562477998]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.44',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91558941773444, 40.83428380733851]},
       'geometry_name': 'geom',
       'properties': {'name': 'Concourse',
        'stacked': 1,
        'annoline1': 'Concourse',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.91558941773444,
         40.83428380733851,
         -73.91558941773444,
         40.83428380733851]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.45',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85053524451935, 40.82977429787161]},
       'geometry_name': 'geom',
       'properties': {'name': 'Unionport',
        'stacked': 1,
        'annoline1': 'Unionport',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.85053524451935,
         40.82977429787161,
         -73.85053524451935,
         40.82977429787161]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.46',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84808271877168, 40.88456130303732]},
       'geometry_name': 'geom',
       'properties': {'name': 'Edenwald',
        'stacked': 1,
        'annoline1': 'Edenwald',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.84808271877168,
         40.88456130303732,
         -73.84808271877168,
         40.88456130303732]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.47',
       'geometry': {'type': 'Point',
        'coordinates': [-74.03062069353813, 40.625801065010656]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bay Ridge',
        'stacked': 1,
        'annoline1': 'Bay Ridge',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-74.03062069353813,
         40.625801065010656,
         -74.03062069353813,
         40.625801065010656]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.48',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99517998380729, 40.61100890202044]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bensonhurst',
        'stacked': 1,
        'annoline1': 'Bensonhurst',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99517998380729,
         40.61100890202044,
         -73.99517998380729,
         40.61100890202044]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.49',
       'geometry': {'type': 'Point',
        'coordinates': [-74.01031618527784, 40.64510294925429]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sunset Park',
        'stacked': 2,
        'annoline1': 'Sunset',
        'annoline2': 'Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-74.01031618527784,
         40.64510294925429,
         -74.01031618527784,
         40.64510294925429]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.50',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95424093127393, 40.7302009848647]},
       'geometry_name': 'geom',
       'properties': {'name': 'Greenpoint',
        'stacked': 1,
        'annoline1': 'Greenpoint',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95424093127393,
         40.7302009848647,
         -73.95424093127393,
         40.7302009848647]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.51',
       'geometry': {'type': 'Point',
        'coordinates': [-73.97347087708445, 40.59526001306593]},
       'geometry_name': 'geom',
       'properties': {'name': 'Gravesend',
        'stacked': 1,
        'annoline1': 'Gravesend',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.97347087708445,
         40.59526001306593,
         -73.97347087708445,
         40.59526001306593]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.52',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96509448785336, 40.57682506566604]},
       'geometry_name': 'geom',
       'properties': {'name': 'Brighton Beach',
        'stacked': 2,
        'annoline1': 'Brighton',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.96509448785336,
         40.57682506566604,
         -73.96509448785336,
         40.57682506566604]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.53',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94318640482979, 40.58689012678384]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sheepshead Bay',
        'stacked': 2,
        'annoline1': 'Sheepshead',
        'annoline2': 'Bay',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.94318640482979,
         40.58689012678384,
         -73.94318640482979,
         40.58689012678384]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.54',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95743840559939, 40.61443251335098]},
       'geometry_name': 'geom',
       'properties': {'name': 'Manhattan Terrace',
        'stacked': 2,
        'annoline1': 'Manhattan',
        'annoline2': 'Terrace',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95743840559939,
         40.61443251335098,
         -73.95743840559939,
         40.61443251335098]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.55',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95840106533903, 40.63632589026677]},
       'geometry_name': 'geom',
       'properties': {'name': 'Flatbush',
        'stacked': 1,
        'annoline1': 'Flatbush',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95840106533903,
         40.63632589026677,
         -73.95840106533903,
         40.63632589026677]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.56',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94329119073582, 40.67082917695294]},
       'geometry_name': 'geom',
       'properties': {'name': 'Crown Heights',
        'stacked': 2,
        'annoline1': 'Crown',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.94329119073582,
         40.67082917695294,
         -73.94329119073582,
         40.67082917695294]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.57',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93610256185836, 40.64171776668961]},
       'geometry_name': 'geom',
       'properties': {'name': 'East Flatbush',
        'stacked': 1,
        'annoline1': 'East Flatbush',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.93610256185836,
         40.64171776668961,
         -73.93610256185836,
         40.64171776668961]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.58',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98042110559474, 40.642381958003526]},
       'geometry_name': 'geom',
       'properties': {'name': 'Kensington',
        'stacked': 1,
        'annoline1': 'Kensington',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.98042110559474,
         40.642381958003526,
         -73.98042110559474,
         40.642381958003526]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.59',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98007340430172, 40.65694583575104]},
       'geometry_name': 'geom',
       'properties': {'name': 'Windsor Terrace',
        'stacked': 2,
        'annoline1': 'Windsor',
        'annoline2': 'Terrace',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.98007340430172,
         40.65694583575104,
         -73.98007340430172,
         40.65694583575104]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.60',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9648592426269, 40.676822262254724]},
       'geometry_name': 'geom',
       'properties': {'name': 'Prospect Heights',
        'stacked': 2,
        'annoline1': 'Prospect',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.9648592426269,
         40.676822262254724,
         -73.9648592426269,
         40.676822262254724]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.61',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91023536176607, 40.66394994339755]},
       'geometry_name': 'geom',
       'properties': {'name': 'Brownsville',
        'stacked': 1,
        'annoline1': 'Brownsville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.91023536176607,
         40.66394994339755,
         -73.91023536176607,
         40.66394994339755]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.62',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95811529220927, 40.70714439344251]},
       'geometry_name': 'geom',
       'properties': {'name': 'Williamsburg',
        'stacked': 1,
        'annoline1': 'Williamsburg',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95811529220927,
         40.70714439344251,
         -73.95811529220927,
         40.70714439344251]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.63',
       'geometry': {'type': 'Point',
        'coordinates': [-73.92525797487045, 40.69811611017901]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bushwick',
        'stacked': 1,
        'annoline1': 'Bushwick',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.92525797487045,
         40.69811611017901,
         -73.92525797487045,
         40.69811611017901]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.64',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94178488690297, 40.687231607720456]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bedford Stuyvesant',
        'stacked': 1,
        'annoline1': 'Bedford Stuyvesant',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.94178488690297,
         40.687231607720456,
         -73.94178488690297,
         40.687231607720456]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.65',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99378225496424, 40.695863722724084]},
       'geometry_name': 'geom',
       'properties': {'name': 'Brooklyn Heights',
        'stacked': 2,
        'annoline1': 'Brooklyn',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99378225496424,
         40.695863722724084,
         -73.99378225496424,
         40.695863722724084]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.66',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99856139218463, 40.687919722485574]},
       'geometry_name': 'geom',
       'properties': {'name': 'Cobble Hill',
        'stacked': 2,
        'annoline1': 'Cobble',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99856139218463,
         40.687919722485574,
         -73.99856139218463,
         40.687919722485574]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.67',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99465372828006, 40.680540231076485]},
       'geometry_name': 'geom',
       'properties': {'name': 'Carroll Gardens',
        'stacked': 2,
        'annoline1': 'Carroll',
        'annoline2': 'Gardens',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99465372828006,
         40.680540231076485,
         -73.99465372828006,
         40.680540231076485]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.68',
       'geometry': {'type': 'Point',
        'coordinates': [-74.0127589747356, 40.676253230250886]},
       'geometry_name': 'geom',
       'properties': {'name': 'Red Hook',
        'stacked': 2,
        'annoline1': 'Red',
        'annoline2': 'Hook',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-74.0127589747356,
         40.676253230250886,
         -74.0127589747356,
         40.676253230250886]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.69',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99444087145339, 40.673931143187154]},
       'geometry_name': 'geom',
       'properties': {'name': 'Gowanus',
        'stacked': 1,
        'annoline1': 'Gowanus',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99444087145339,
         40.673931143187154,
         -73.99444087145339,
         40.673931143187154]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.70',
       'geometry': {'type': 'Point',
        'coordinates': [-73.97290574369092, 40.68852726018977]},
       'geometry_name': 'geom',
       'properties': {'name': 'Fort Greene',
        'stacked': 2,
        'annoline1': 'Fort',
        'annoline2': 'Greene',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.97290574369092,
         40.68852726018977,
         -73.97290574369092,
         40.68852726018977]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.71',
       'geometry': {'type': 'Point',
        'coordinates': [-73.97705030183924, 40.67232052268197]},
       'geometry_name': 'geom',
       'properties': {'name': 'Park Slope',
        'stacked': 2,
        'annoline1': 'Park',
        'annoline2': 'Slope',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.97705030183924,
         40.67232052268197,
         -73.97705030183924,
         40.67232052268197]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.72',
       'geometry': {'type': 'Point',
        'coordinates': [-73.87661596457296, 40.68239101144211]},
       'geometry_name': 'geom',
       'properties': {'name': 'Cypress Hills',
        'stacked': 2,
        'annoline1': 'Cypress',
        'annoline2': 'Hills',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.87661596457296,
         40.68239101144211,
         -73.87661596457296,
         40.68239101144211]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.73',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88069863917366, 40.669925700847045]},
       'geometry_name': 'geom',
       'properties': {'name': 'East New York',
        'stacked': 1,
        'annoline1': 'East New York',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.88069863917366,
         40.669925700847045,
         -73.88069863917366,
         40.669925700847045]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.74',
       'geometry': {'type': 'Point',
        'coordinates': [-73.87936970045875, 40.64758905230874]},
       'geometry_name': 'geom',
       'properties': {'name': 'Starrett City',
        'stacked': 2,
        'annoline1': 'Starrett',
        'annoline2': 'City',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.87936970045875,
         40.64758905230874,
         -73.87936970045875,
         40.64758905230874]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.75',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90209269778966, 40.63556432797428]},
       'geometry_name': 'geom',
       'properties': {'name': 'Canarsie',
        'stacked': 1,
        'annoline1': 'Canarsie',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.90209269778966,
         40.63556432797428,
         -73.90209269778966,
         40.63556432797428]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.76',
       'geometry': {'type': 'Point',
        'coordinates': [-73.92911302644674, 40.630446043757466]},
       'geometry_name': 'geom',
       'properties': {'name': 'Flatlands',
        'stacked': 1,
        'annoline1': 'Flatlands',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.92911302644674,
         40.630446043757466,
         -73.92911302644674,
         40.630446043757466]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.77',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90818571777423, 40.606336421685626]},
       'geometry_name': 'geom',
       'properties': {'name': 'Mill Island',
        'stacked': 2,
        'annoline1': 'Mill',
        'annoline2': 'Island',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.90818571777423,
         40.606336421685626,
         -73.90818571777423,
         40.606336421685626]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.78',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94353722891886, 40.57791350308657]},
       'geometry_name': 'geom',
       'properties': {'name': 'Manhattan Beach',
        'stacked': 2,
        'annoline1': 'Manhattan',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.94353722891886,
         40.57791350308657,
         -73.94353722891886,
         40.57791350308657]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.79',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98868295821637, 40.57429256471601]},
       'geometry_name': 'geom',
       'properties': {'name': 'Coney Island',
        'stacked': 1,
        'annoline1': 'Coney Island',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.98868295821637,
         40.57429256471601,
         -73.98868295821637,
         40.57429256471601]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.80',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99875221443519, 40.59951870282238]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bath Beach',
        'stacked': 2,
        'annoline1': 'Bath',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99875221443519,
         40.59951870282238,
         -73.99875221443519,
         40.59951870282238]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.81',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99049823044811, 40.633130512758015]},
       'geometry_name': 'geom',
       'properties': {'name': 'Borough Park',
        'stacked': 2,
        'annoline1': 'Borough',
        'annoline2': 'Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99049823044811,
         40.633130512758015,
         -73.99049823044811,
         40.633130512758015]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.82',
       'geometry': {'type': 'Point',
        'coordinates': [-74.01931375636022, 40.619219457722636]},
       'geometry_name': 'geom',
       'properties': {'name': 'Dyker Heights',
        'stacked': 2,
        'annoline1': 'Dyker',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-74.01931375636022,
         40.619219457722636,
         -74.01931375636022,
         40.619219457722636]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.83',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93010170691196, 40.590848433902046]},
       'geometry_name': 'geom',
       'properties': {'name': 'Gerritsen Beach',
        'stacked': 2,
        'annoline1': 'Gerritsen',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.93010170691196,
         40.590848433902046,
         -73.93010170691196,
         40.590848433902046]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.84',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93134404108497, 40.609747779894604]},
       'geometry_name': 'geom',
       'properties': {'name': 'Marine Park',
        'stacked': 1,
        'annoline1': 'Marine Park',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.93134404108497,
         40.609747779894604,
         -73.93134404108497,
         40.609747779894604]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.85',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96784306216367, 40.693229421881504]},
       'geometry_name': 'geom',
       'properties': {'name': 'Clinton Hill',
        'stacked': 2,
        'annoline1': 'Clinton',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.96784306216367,
         40.693229421881504,
         -73.96784306216367,
         40.693229421881504]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.86',
       'geometry': {'type': 'Point',
        'coordinates': [-74.0078731120024, 40.57637537890224]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sea Gate',
        'stacked': 2,
        'annoline1': 'Sea',
        'annoline2': 'Gate',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-74.0078731120024,
         40.57637537890224,
         -74.0078731120024,
         40.57637537890224]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.87',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98346337431099, 40.69084402109802]},
       'geometry_name': 'geom',
       'properties': {'name': 'Downtown',
        'stacked': 1,
        'annoline1': 'Downtown',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.98346337431099,
         40.69084402109802,
         -73.98346337431099,
         40.69084402109802]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.88',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98374824115798, 40.685682912091444]},
       'geometry_name': 'geom',
       'properties': {'name': 'Boerum Hill',
        'stacked': 2,
        'annoline1': 'Boerum',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.98374824115798,
         40.685682912091444,
         -73.98374824115798,
         40.685682912091444]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.89',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95489867077713, 40.658420017469815]},
       'geometry_name': 'geom',
       'properties': {'name': 'Prospect Lefferts Gardens',
        'stacked': 3,
        'annoline1': 'Prospect',
        'annoline2': 'Lefferts',
        'annoline3': 'Gardens',
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95489867077713,
         40.658420017469815,
         -73.95489867077713,
         40.658420017469815]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.90',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91306831787395, 40.678402554795355]},
       'geometry_name': 'geom',
       'properties': {'name': 'Ocean Hill',
        'stacked': 2,
        'annoline1': 'Ocean',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.91306831787395,
         40.678402554795355,
         -73.91306831787395,
         40.678402554795355]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.91',
       'geometry': {'type': 'Point',
        'coordinates': [-73.86797598081334, 40.67856995727479]},
       'geometry_name': 'geom',
       'properties': {'name': 'City Line',
        'stacked': 2,
        'annoline1': 'City',
        'annoline2': 'Line',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.86797598081334,
         40.67856995727479,
         -73.86797598081334,
         40.67856995727479]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.92',
       'geometry': {'type': 'Point',
        'coordinates': [-73.89855633630317, 40.61514955045308]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bergen Beach',
        'stacked': 2,
        'annoline1': 'Bergen',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.89855633630317,
         40.61514955045308,
         -73.89855633630317,
         40.61514955045308]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.93',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95759523489838, 40.62559589869843]},
       'geometry_name': 'geom',
       'properties': {'name': 'Midwood',
        'stacked': 1,
        'annoline1': 'Midwood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95759523489838,
         40.62559589869843,
         -73.95759523489838,
         40.62559589869843]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.94',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96261316716048, 40.647008603185185]},
       'geometry_name': 'geom',
       'properties': {'name': 'Prospect Park South',
        'stacked': 2,
        'annoline1': 'Prospect',
        'annoline2': 'Park South',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.96261316716048,
         40.647008603185185,
         -73.96261316716048,
         40.647008603185185]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.95',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91607483951324, 40.62384524478419]},
       'geometry_name': 'geom',
       'properties': {'name': 'Georgetown',
        'stacked': 1,
        'annoline1': 'Georgetown',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.91607483951324,
         40.62384524478419,
         -73.91607483951324,
         40.62384524478419]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.96',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93885815269195, 40.70849241041548]},
       'geometry_name': 'geom',
       'properties': {'name': 'East Williamsburg',
        'stacked': 2,
        'annoline1': 'East',
        'annoline2': 'Williamsburg',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.93885815269195,
         40.70849241041548,
         -73.93885815269195,
         40.70849241041548]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.97',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95880857587582, 40.714822906532014]},
       'geometry_name': 'geom',
       'properties': {'name': 'North Side',
        'stacked': 1,
        'annoline1': 'North Side',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95880857587582,
         40.714822906532014,
         -73.95880857587582,
         40.714822906532014]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.98',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95800095153331, 40.71086147265064]},
       'geometry_name': 'geom',
       'properties': {'name': 'South Side',
        'stacked': 1,
        'annoline1': 'South Side',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95800095153331,
         40.71086147265064,
         -73.95800095153331,
         40.71086147265064]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.99',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96836678035541, 40.61305976667942]},
       'geometry_name': 'geom',
       'properties': {'name': 'Ocean Parkway',
        'stacked': 2,
        'annoline1': 'Ocean',
        'annoline2': 'Parkway',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.96836678035541,
         40.61305976667942,
         -73.96836678035541,
         40.61305976667942]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.100',
       'geometry': {'type': 'Point',
        'coordinates': [-74.03197914537984, 40.61476812694226]},
       'geometry_name': 'geom',
       'properties': {'name': 'Fort Hamilton',
        'stacked': 2,
        'annoline1': 'Fort',
        'annoline2': 'Hamilton',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-74.03197914537984,
         40.61476812694226,
         -74.03197914537984,
         40.61476812694226]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.101',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99427936255978, 40.71561842231432]},
       'geometry_name': 'geom',
       'properties': {'name': 'Chinatown',
        'stacked': 1,
        'annoline1': 'Chinatown',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.99427936255978,
         40.71561842231432,
         -73.99427936255978,
         40.71561842231432]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.102',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93690027985234, 40.85190252555305]},
       'geometry_name': 'geom',
       'properties': {'name': 'Washington Heights',
        'stacked': 2,
        'annoline1': 'Washington',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.93690027985234,
         40.85190252555305,
         -73.93690027985234,
         40.85190252555305]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.103',
       'geometry': {'type': 'Point',
        'coordinates': [-73.92121042203897, 40.86768396449915]},
       'geometry_name': 'geom',
       'properties': {'name': 'Inwood',
        'stacked': 1,
        'annoline1': 'Inwood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.92121042203897,
         40.86768396449915,
         -73.92121042203897,
         40.86768396449915]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.104',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94968791883366, 40.823604284811935]},
       'geometry_name': 'geom',
       'properties': {'name': 'Hamilton Heights',
        'stacked': 2,
        'annoline1': 'Hamilton',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.94968791883366,
         40.823604284811935,
         -73.94968791883366,
         40.823604284811935]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.105',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9573853935188, 40.8169344294978]},
       'geometry_name': 'geom',
       'properties': {'name': 'Manhattanville',
        'stacked': 2,
        'annoline1': 'Manhattanville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.9573853935188,
         40.8169344294978,
         -73.9573853935188,
         40.8169344294978]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.106',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94321112603905, 40.81597606742414]},
       'geometry_name': 'geom',
       'properties': {'name': 'Central Harlem',
        'stacked': 2,
        'annoline1': 'Central',
        'annoline2': 'Harlem',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.94321112603905,
         40.81597606742414,
         -73.94321112603905,
         40.81597606742414]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.107',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94418223148524, 40.79224946663033]},
       'geometry_name': 'geom',
       'properties': {'name': 'East Harlem',
        'stacked': 2,
        'annoline1': 'East',
        'annoline2': 'Harlem',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.94418223148524,
         40.79224946663033,
         -73.94418223148524,
         40.79224946663033]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.108',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96050763135, 40.775638573301805]},
       'geometry_name': 'geom',
       'properties': {'name': 'Upper East Side',
        'stacked': 3,
        'annoline1': 'Upper',
        'annoline2': 'East',
        'annoline3': 'Side',
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.96050763135,
         40.775638573301805,
         -73.96050763135,
         40.775638573301805]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.109',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94711784471826, 40.775929849884875]},
       'geometry_name': 'geom',
       'properties': {'name': 'Yorkville',
        'stacked': 1,
        'annoline1': 'Yorkville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.94711784471826,
         40.775929849884875,
         -73.94711784471826,
         40.775929849884875]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.110',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9588596881376, 40.76811265828733]},
       'geometry_name': 'geom',
       'properties': {'name': 'Lenox Hill',
        'stacked': 2,
        'annoline1': 'Lenox',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.9588596881376,
         40.76811265828733,
         -73.9588596881376,
         40.76811265828733]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.111',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94916769227953, 40.76215960576283]},
       'geometry_name': 'geom',
       'properties': {'name': 'Roosevelt Island',
        'stacked': 1,
        'annoline1': 'Roosevelt Island',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 56,
        'borough': 'Manhattan',
        'bbox': [-73.94916769227953,
         40.76215960576283,
         -73.94916769227953,
         40.76215960576283]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.112',
       'geometry': {'type': 'Point',
        'coordinates': [-73.97705923630603, 40.787657998534854]},
       'geometry_name': 'geom',
       'properties': {'name': 'Upper West Side',
        'stacked': 3,
        'annoline1': 'Upper',
        'annoline2': 'West',
        'annoline3': 'Side',
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.97705923630603,
         40.787657998534854,
         -73.97705923630603,
         40.787657998534854]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.113',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98533777001262, 40.77352888942166]},
       'geometry_name': 'geom',
       'properties': {'name': 'Lincoln Square',
        'stacked': 2,
        'annoline1': 'Lincoln',
        'annoline2': 'Square',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.98533777001262,
         40.77352888942166,
         -73.98533777001262,
         40.77352888942166]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.114',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99611936309479, 40.75910089146212]},
       'geometry_name': 'geom',
       'properties': {'name': 'Clinton',
        'stacked': 1,
        'annoline1': 'Clinton',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.99611936309479,
         40.75910089146212,
         -73.99611936309479,
         40.75910089146212]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.115',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98166882730304, 40.75469110270623]},
       'geometry_name': 'geom',
       'properties': {'name': 'Midtown',
        'stacked': 1,
        'annoline1': 'Midtown',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.98166882730304,
         40.75469110270623,
         -73.98166882730304,
         40.75469110270623]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.116',
       'geometry': {'type': 'Point',
        'coordinates': [-73.97833207924127, 40.748303077252174]},
       'geometry_name': 'geom',
       'properties': {'name': 'Murray Hill',
        'stacked': 2,
        'annoline1': 'Murray',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.97833207924127,
         40.748303077252174,
         -73.97833207924127,
         40.748303077252174]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.117',
       'geometry': {'type': 'Point',
        'coordinates': [-74.00311633472813, 40.744034706747975]},
       'geometry_name': 'geom',
       'properties': {'name': 'Chelsea',
        'stacked': 1,
        'annoline1': 'Chelsea',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.00311633472813,
         40.744034706747975,
         -74.00311633472813,
         40.744034706747975]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.118',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99991402945902, 40.72693288536128]},
       'geometry_name': 'geom',
       'properties': {'name': 'Greenwich Village',
        'stacked': 2,
        'annoline1': 'Greenwich',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.99991402945902,
         40.72693288536128,
         -73.99991402945902,
         40.72693288536128]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.119',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98222616506416, 40.727846777270244]},
       'geometry_name': 'geom',
       'properties': {'name': 'East Village',
        'stacked': 2,
        'annoline1': 'East',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.98222616506416,
         40.727846777270244,
         -73.98222616506416,
         40.727846777270244]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.120',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98089031999291, 40.71780674892765]},
       'geometry_name': 'geom',
       'properties': {'name': 'Lower East Side',
        'stacked': 3,
        'annoline1': 'Lower',
        'annoline2': 'East',
        'annoline3': 'Side',
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.98089031999291,
         40.71780674892765,
         -73.98089031999291,
         40.71780674892765]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.121',
       'geometry': {'type': 'Point',
        'coordinates': [-74.01068328559087, 40.721521967443216]},
       'geometry_name': 'geom',
       'properties': {'name': 'Tribeca',
        'stacked': 1,
        'annoline1': 'Tribeca',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.01068328559087,
         40.721521967443216,
         -74.01068328559087,
         40.721521967443216]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.122',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99730467208073, 40.71932379395907]},
       'geometry_name': 'geom',
       'properties': {'name': 'Little Italy',
        'stacked': 2,
        'annoline1': 'Little',
        'annoline2': 'Italy',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.99730467208073,
         40.71932379395907,
         -73.99730467208073,
         40.71932379395907]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.123',
       'geometry': {'type': 'Point',
        'coordinates': [-74.00065666959759, 40.72218384131794]},
       'geometry_name': 'geom',
       'properties': {'name': 'Soho',
        'stacked': 1,
        'annoline1': 'Soho',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.00065666959759,
         40.72218384131794,
         -74.00065666959759,
         40.72218384131794]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.124',
       'geometry': {'type': 'Point',
        'coordinates': [-74.00617998126812, 40.73443393572434]},
       'geometry_name': 'geom',
       'properties': {'name': 'West Village',
        'stacked': 2,
        'annoline1': 'West',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.00617998126812,
         40.73443393572434,
         -74.00617998126812,
         40.73443393572434]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.125',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96428617740655, 40.797307041702865]},
       'geometry_name': 'geom',
       'properties': {'name': 'Manhattan Valley',
        'stacked': 2,
        'annoline1': 'Manhattan',
        'annoline2': 'Valley',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.96428617740655,
         40.797307041702865,
         -73.96428617740655,
         40.797307041702865]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.126',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96389627905332, 40.807999738165826]},
       'geometry_name': 'geom',
       'properties': {'name': 'Morningside Heights',
        'stacked': 2,
        'annoline1': 'Morningside',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.96389627905332,
         40.807999738165826,
         -73.96389627905332,
         40.807999738165826]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.127',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98137594833541, 40.737209832715]},
       'geometry_name': 'geom',
       'properties': {'name': 'Gramercy',
        'stacked': 1,
        'annoline1': 'Gramercy',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.98137594833541,
         40.737209832715,
         -73.98137594833541,
         40.737209832715]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.128',
       'geometry': {'type': 'Point',
        'coordinates': [-74.01686930508617, 40.71193198394565]},
       'geometry_name': 'geom',
       'properties': {'name': 'Battery Park City',
        'stacked': 3,
        'annoline1': 'Battery',
        'annoline2': 'Park',
        'annoline3': 'City',
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.01686930508617,
         40.71193198394565,
         -74.01686930508617,
         40.71193198394565]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.129',
       'geometry': {'type': 'Point',
        'coordinates': [-74.0106654452127, 40.70710710727048]},
       'geometry_name': 'geom',
       'properties': {'name': 'Financial District',
        'stacked': 2,
        'annoline1': 'Financial',
        'annoline2': 'District',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.0106654452127,
         40.70710710727048,
         -74.0106654452127,
         40.70710710727048]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.130',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91565374304234, 40.76850859335492]},
       'geometry_name': 'geom',
       'properties': {'name': 'Astoria',
        'stacked': 1,
        'annoline1': 'Astoria',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.91565374304234,
         40.76850859335492,
         -73.91565374304234,
         40.76850859335492]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.131',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90184166838284, 40.74634908860222]},
       'geometry_name': 'geom',
       'properties': {'name': 'Woodside',
        'stacked': 1,
        'annoline1': 'Woodside',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.90184166838284,
         40.74634908860222,
         -73.90184166838284,
         40.74634908860222]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.132',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88282109164365, 40.75198138007367]},
       'geometry_name': 'geom',
       'properties': {'name': 'Jackson Heights',
        'stacked': 2,
        'annoline1': 'Jackson',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.88282109164365,
         40.75198138007367,
         -73.88282109164365,
         40.75198138007367]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.133',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88165622288388, 40.744048505122024]},
       'geometry_name': 'geom',
       'properties': {'name': 'Elmhurst',
        'stacked': 1,
        'annoline1': 'Elmhurst',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.88165622288388,
         40.744048505122024,
         -73.88165622288388,
         40.744048505122024]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.134',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8381376460028, 40.65422527738487]},
       'geometry_name': 'geom',
       'properties': {'name': 'Howard Beach',
        'stacked': 2,
        'annoline1': 'Howard',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.8381376460028,
         40.65422527738487,
         -73.8381376460028,
         40.65422527738487]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.135',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85682497345258, 40.74238175015667]},
       'geometry_name': 'geom',
       'properties': {'name': 'Corona',
        'stacked': 1,
        'annoline1': 'Corona',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.85682497345258,
         40.74238175015667,
         -73.85682497345258,
         40.74238175015667]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.136',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84447500788983, 40.72526378216503]},
       'geometry_name': 'geom',
       'properties': {'name': 'Forest Hills',
        'stacked': 2,
        'annoline1': 'Forest',
        'annoline2': 'Hills',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.84447500788983,
         40.72526378216503,
         -73.84447500788983,
         40.72526378216503]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.137',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82981905825703, 40.7051790354148]},
       'geometry_name': 'geom',
       'properties': {'name': 'Kew Gardens',
        'stacked': 2,
        'annoline1': 'Kew',
        'annoline2': 'Gardens',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.82981905825703,
         40.7051790354148,
         -73.82981905825703,
         40.7051790354148]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.138',
       'geometry': {'type': 'Point',
        'coordinates': [-73.83183321446887, 40.69794731471763]},
       'geometry_name': 'geom',
       'properties': {'name': 'Richmond Hill',
        'stacked': 2,
        'annoline1': 'Richmond',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.83183321446887,
         40.69794731471763,
         -73.83183321446887,
         40.69794731471763]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.139',
       'geometry': {'type': 'Point',
        'coordinates': [-73.83177300329582, 40.76445419697846]},
       'geometry_name': 'geom',
       'properties': {'name': 'Flushing',
        'stacked': 1,
        'annoline1': 'Flushing',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.83177300329582,
         40.76445419697846,
         -73.83177300329582,
         40.76445419697846]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.140',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93920223915505, 40.75021734610528]},
       'geometry_name': 'geom',
       'properties': {'name': 'Long Island City',
        'stacked': 3,
        'annoline1': 'Long',
        'annoline2': 'Island',
        'annoline3': 'City',
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.93920223915505,
         40.75021734610528,
         -73.93920223915505,
         40.75021734610528]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.141',
       'geometry': {'type': 'Point',
        'coordinates': [-73.92691617561577, 40.74017628351924]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sunnyside',
        'stacked': 1,
        'annoline1': 'Sunnyside',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.92691617561577,
         40.74017628351924,
         -73.92691617561577,
         40.74017628351924]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.142',
       'geometry': {'type': 'Point',
        'coordinates': [-73.86704147658772, 40.76407323883091]},
       'geometry_name': 'geom',
       'properties': {'name': 'East Elmhurst',
        'stacked': 2,
        'annoline1': 'East',
        'annoline2': 'Elmhurst',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.86704147658772,
         40.76407323883091,
         -73.86704147658772,
         40.76407323883091]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.143',
       'geometry': {'type': 'Point',
        'coordinates': [-73.89621713626859, 40.725427374093606]},
       'geometry_name': 'geom',
       'properties': {'name': 'Maspeth',
        'stacked': 1,
        'annoline1': 'Maspeth',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.89621713626859,
         40.725427374093606,
         -73.89621713626859,
         40.725427374093606]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.144',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90143517559589, 40.70832315613858]},
       'geometry_name': 'geom',
       'properties': {'name': 'Ridgewood',
        'stacked': 1,
        'annoline1': 'Ridgewood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.90143517559589,
         40.70832315613858,
         -73.90143517559589,
         40.70832315613858]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.145',
       'geometry': {'type': 'Point',
        'coordinates': [-73.87074167435605, 40.70276242967838]},
       'geometry_name': 'geom',
       'properties': {'name': 'Glendale',
        'stacked': 1,
        'annoline1': 'Glendale',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.87074167435605,
         40.70276242967838,
         -73.87074167435605,
         40.70276242967838]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.146',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8578268690537, 40.72897409480735]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rego Park',
        'stacked': 1,
        'annoline1': 'Rego Park',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.8578268690537,
         40.72897409480735,
         -73.8578268690537,
         40.72897409480735]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.147',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8581104655432, 40.68988687915789]},
       'geometry_name': 'geom',
       'properties': {'name': 'Woodhaven',
        'stacked': 1,
        'annoline1': 'Woodhaven',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.8581104655432,
         40.68988687915789,
         -73.8581104655432,
         40.68988687915789]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.148',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84320266173447, 40.680708468265415]},
       'geometry_name': 'geom',
       'properties': {'name': 'Ozone Park',
        'stacked': 1,
        'annoline1': 'Ozone Park',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.84320266173447,
         40.680708468265415,
         -73.84320266173447,
         40.680708468265415]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.149',
       'geometry': {'type': 'Point',
        'coordinates': [-73.80986478649041, 40.66854957767195]},
       'geometry_name': 'geom',
       'properties': {'name': 'South Ozone Park',
        'stacked': 2,
        'annoline1': 'South',
        'annoline2': 'Ozone Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.80986478649041,
         40.66854957767195,
         -73.80986478649041,
         40.66854957767195]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.150',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84304528896125, 40.784902749260205]},
       'geometry_name': 'geom',
       'properties': {'name': 'College Point',
        'stacked': 2,
        'annoline1': 'College',
        'annoline2': 'Point',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.84304528896125,
         40.784902749260205,
         -73.84304528896125,
         40.784902749260205]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.151',
       'geometry': {'type': 'Point',
        'coordinates': [-73.81420216610863, 40.78129076602694]},
       'geometry_name': 'geom',
       'properties': {'name': 'Whitestone',
        'stacked': 1,
        'annoline1': 'Whitestone',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.81420216610863,
         40.78129076602694,
         -73.81420216610863,
         40.78129076602694]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.152',
       'geometry': {'type': 'Point',
        'coordinates': [-73.7742736306867, 40.76604063281064]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bayside',
        'stacked': 1,
        'annoline1': 'Bayside',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.7742736306867,
         40.76604063281064,
         -73.7742736306867,
         40.76604063281064]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.153',
       'geometry': {'type': 'Point',
        'coordinates': [-73.79176243728061, 40.76172954903262]},
       'geometry_name': 'geom',
       'properties': {'name': 'Auburndale',
        'stacked': 1,
        'annoline1': 'Auburndale',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.79176243728061,
         40.76172954903262,
         -73.79176243728061,
         40.76172954903262]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.154',
       'geometry': {'type': 'Point',
        'coordinates': [-73.7388977558074, 40.7708261928267]},
       'geometry_name': 'geom',
       'properties': {'name': 'Little Neck',
        'stacked': 2,
        'annoline1': 'Little',
        'annoline2': 'Neck',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.7388977558074,
         40.7708261928267,
         -73.7388977558074,
         40.7708261928267]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.155',
       'geometry': {'type': 'Point',
        'coordinates': [-73.7424982072733, 40.76684609790763]},
       'geometry_name': 'geom',
       'properties': {'name': 'Douglaston',
        'stacked': 1,
        'annoline1': 'Douglaston',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.7424982072733,
         40.76684609790763,
         -73.7424982072733,
         40.76684609790763]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.156',
       'geometry': {'type': 'Point',
        'coordinates': [-73.71548118999145, 40.74944079974332]},
       'geometry_name': 'geom',
       'properties': {'name': 'Glen Oaks',
        'stacked': 2,
        'annoline1': 'Glen',
        'annoline2': 'Oaks',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.71548118999145,
         40.74944079974332,
         -73.71548118999145,
         40.74944079974332]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.157',
       'geometry': {'type': 'Point',
        'coordinates': [-73.72012814826903, 40.72857318176675]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bellerose',
        'stacked': 1,
        'annoline1': 'Bellerose',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.72012814826903,
         40.72857318176675,
         -73.72012814826903,
         40.72857318176675]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.158',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82087764933566, 40.722578244228046]},
       'geometry_name': 'geom',
       'properties': {'name': 'Kew Gardens Hills',
        'stacked': 3,
        'annoline1': 'Kew',
        'annoline2': 'Gardens',
        'annoline3': 'Hills',
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.82087764933566,
         40.722578244228046,
         -73.82087764933566,
         40.722578244228046]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.159',
       'geometry': {'type': 'Point',
        'coordinates': [-73.78271337003264, 40.7343944653313]},
       'geometry_name': 'geom',
       'properties': {'name': 'Fresh Meadows',
        'stacked': 2,
        'annoline1': 'Fresh',
        'annoline2': 'Meadows',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.78271337003264,
         40.7343944653313,
         -73.78271337003264,
         40.7343944653313]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.160',
       'geometry': {'type': 'Point',
        'coordinates': [-73.81174822458634, 40.71093547252271]},
       'geometry_name': 'geom',
       'properties': {'name': 'Briarwood',
        'stacked': 1,
        'annoline1': 'Briarwood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.81174822458634,
         40.71093547252271,
         -73.81174822458634,
         40.71093547252271]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.161',
       'geometry': {'type': 'Point',
        'coordinates': [-73.79690165888289, 40.70465736068717]},
       'geometry_name': 'geom',
       'properties': {'name': 'Jamaica Center',
        'stacked': 2,
        'annoline1': 'Jamaica',
        'annoline2': 'Center',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.79690165888289,
         40.70465736068717,
         -73.79690165888289,
         40.70465736068717]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.162',
       'geometry': {'type': 'Point',
        'coordinates': [-73.75494976234332, 40.74561857141855]},
       'geometry_name': 'geom',
       'properties': {'name': 'Oakland Gardens',
        'stacked': 2,
        'annoline1': 'Oakland',
        'annoline2': 'Gardens',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.75494976234332,
         40.74561857141855,
         -73.75494976234332,
         40.74561857141855]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.163',
       'geometry': {'type': 'Point',
        'coordinates': [-73.73871484578424, 40.718893092167356]},
       'geometry_name': 'geom',
       'properties': {'name': 'Queens Village',
        'stacked': 2,
        'annoline1': 'Queens',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.73871484578424,
         40.718893092167356,
         -73.73871484578424,
         40.718893092167356]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.164',
       'geometry': {'type': 'Point',
        'coordinates': [-73.75925009335594, 40.71124344191904]},
       'geometry_name': 'geom',
       'properties': {'name': 'Hollis',
        'stacked': 1,
        'annoline1': 'Hollis',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.75925009335594,
         40.71124344191904,
         -73.75925009335594,
         40.71124344191904]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.165',
       'geometry': {'type': 'Point',
        'coordinates': [-73.7904261313554, 40.696911253789885]},
       'geometry_name': 'geom',
       'properties': {'name': 'South Jamaica',
        'stacked': 1,
        'annoline1': 'South Jamaica',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.7904261313554,
         40.696911253789885,
         -73.7904261313554,
         40.696911253789885]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.166',
       'geometry': {'type': 'Point',
        'coordinates': [-73.75867603727717, 40.69444538522359]},
       'geometry_name': 'geom',
       'properties': {'name': 'St. Albans',
        'stacked': 1,
        'annoline1': 'St. Albans',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.75867603727717,
         40.69444538522359,
         -73.75867603727717,
         40.69444538522359]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.167',
       'geometry': {'type': 'Point',
        'coordinates': [-73.77258787620906, 40.67521139591733]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rochdale',
        'stacked': 1,
        'annoline1': 'Rochdale',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.77258787620906,
         40.67521139591733,
         -73.77258787620906,
         40.67521139591733]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.168',
       'geometry': {'type': 'Point',
        'coordinates': [-73.76042092682287, 40.666230490368584]},
       'geometry_name': 'geom',
       'properties': {'name': 'Springfield Gardens',
        'stacked': 2,
        'annoline1': 'Springfield',
        'annoline2': 'Gardens',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.76042092682287,
         40.666230490368584,
         -73.76042092682287,
         40.666230490368584]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.169',
       'geometry': {'type': 'Point',
        'coordinates': [-73.73526873708026, 40.692774639160845]},
       'geometry_name': 'geom',
       'properties': {'name': 'Cambria Heights',
        'stacked': 2,
        'annoline1': 'Cambria',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.73526873708026,
         40.692774639160845,
         -73.73526873708026,
         40.692774639160845]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.170',
       'geometry': {'type': 'Point',
        'coordinates': [-73.73526079428278, 40.659816433428084]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rosedale',
        'stacked': 1,
        'annoline1': 'Rosedale',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.73526079428278,
         40.659816433428084,
         -73.73526079428278,
         40.659816433428084]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.171',
       'geometry': {'type': 'Point',
        'coordinates': [-73.75497968043872, 40.603134432500894]},
       'geometry_name': 'geom',
       'properties': {'name': 'Far Rockaway',
        'stacked': 2,
        'annoline1': 'Far Rockaway',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.75497968043872,
         40.603134432500894,
         -73.75497968043872,
         40.603134432500894]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.172',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8200548911032, 40.60302658351238]},
       'geometry_name': 'geom',
       'properties': {'name': 'Broad Channel',
        'stacked': 2,
        'annoline1': 'Broad',
        'annoline2': 'Channel',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.8200548911032,
         40.60302658351238,
         -73.8200548911032,
         40.60302658351238]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.173',
       'geometry': {'type': 'Point',
        'coordinates': [-73.92551196994168, 40.55740128845452]},
       'geometry_name': 'geom',
       'properties': {'name': 'Breezy Point',
        'stacked': 2,
        'annoline1': 'Breezy',
        'annoline2': 'Point',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.92551196994168,
         40.55740128845452,
         -73.92551196994168,
         40.55740128845452]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.174',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90228960391673, 40.775923015642896]},
       'geometry_name': 'geom',
       'properties': {'name': 'Steinway',
        'stacked': 1,
        'annoline1': 'Steinway',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.90228960391673,
         40.775923015642896,
         -73.90228960391673,
         40.775923015642896]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.175',
       'geometry': {'type': 'Point',
        'coordinates': [-73.80436451720988, 40.79278140360048]},
       'geometry_name': 'geom',
       'properties': {'name': 'Beechhurst',
        'stacked': 1,
        'annoline1': 'Beechhurst',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.80436451720988,
         40.79278140360048,
         -73.80436451720988,
         40.79278140360048]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.176',
       'geometry': {'type': 'Point',
        'coordinates': [-73.7768022262158, 40.782842806245554]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bay Terrace',
        'stacked': 2,
        'annoline1': 'Bay',
        'annoline2': 'Terrace',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.7768022262158,
         40.782842806245554,
         -73.7768022262158,
         40.782842806245554]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.177',
       'geometry': {'type': 'Point',
        'coordinates': [-73.77613282391705, 40.595641807368494]},
       'geometry_name': 'geom',
       'properties': {'name': 'Edgemere',
        'stacked': 1,
        'annoline1': 'Edgemere',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.77613282391705,
         40.595641807368494,
         -73.77613282391705,
         40.595641807368494]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.178',
       'geometry': {'type': 'Point',
        'coordinates': [-73.79199233136943, 40.58914394372971]},
       'geometry_name': 'geom',
       'properties': {'name': 'Arverne',
        'stacked': 1,
        'annoline1': 'Arverne',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.79199233136943,
         40.58914394372971,
         -73.79199233136943,
         40.58914394372971]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.179',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82236121088751, 40.582801696845586]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rockaway Beach',
        'stacked': 2,
        'annoline1': 'Rockaway',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.82236121088751,
         40.582801696845586,
         -73.82236121088751,
         40.582801696845586]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.180',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85754672410827, 40.572036730217015]},
       'geometry_name': 'geom',
       'properties': {'name': 'Neponsit',
        'stacked': 1,
        'annoline1': 'Neponsit',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.85754672410827,
         40.572036730217015,
         -73.85754672410827,
         40.572036730217015]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.181',
       'geometry': {'type': 'Point',
        'coordinates': [-73.81276269135866, 40.764126122614066]},
       'geometry_name': 'geom',
       'properties': {'name': 'Murray Hill',
        'stacked': 2,
        'annoline1': 'Murray',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.81276269135866,
         40.764126122614066,
         -73.81276269135866,
         40.764126122614066]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.182',
       'geometry': {'type': 'Point',
        'coordinates': [-73.70884705889246, 40.741378421945434]},
       'geometry_name': 'geom',
       'properties': {'name': 'Floral Park',
        'stacked': 1,
        'annoline1': 'Floral Park',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.70884705889246,
         40.741378421945434,
         -73.70884705889246,
         40.741378421945434]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.183',
       'geometry': {'type': 'Point',
        'coordinates': [-73.76714166714729, 40.7209572076444]},
       'geometry_name': 'geom',
       'properties': {'name': 'Holliswood',
        'stacked': 1,
        'annoline1': 'Holliswood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.76714166714729,
         40.7209572076444,
         -73.76714166714729,
         40.7209572076444]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.184',
       'geometry': {'type': 'Point',
        'coordinates': [-73.7872269693666, 40.71680483014613]},
       'geometry_name': 'geom',
       'properties': {'name': 'Jamaica Estates',
        'stacked': 2,
        'annoline1': 'Jamaica',
        'annoline2': 'Estates',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.7872269693666,
         40.71680483014613,
         -73.7872269693666,
         40.71680483014613]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.185',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82580915110559, 40.7445723092867]},
       'geometry_name': 'geom',
       'properties': {'name': 'Queensboro Hill',
        'stacked': 2,
        'annoline1': 'Queensboro',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.82580915110559,
         40.7445723092867,
         -73.82580915110559,
         40.7445723092867]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.186',
       'geometry': {'type': 'Point',
        'coordinates': [-73.79760300912672, 40.723824901829204]},
       'geometry_name': 'geom',
       'properties': {'name': 'Hillcrest',
        'stacked': 1,
        'annoline1': 'Hillcrest',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.79760300912672,
         40.723824901829204,
         -73.79760300912672,
         40.723824901829204]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.187',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93157506072878, 40.761704526054146]},
       'geometry_name': 'geom',
       'properties': {'name': 'Ravenswood',
        'stacked': 1,
        'annoline1': 'Ravenswood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.93157506072878,
         40.761704526054146,
         -73.93157506072878,
         40.761704526054146]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.188',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84963782402441, 40.66391841925139]},
       'geometry_name': 'geom',
       'properties': {'name': 'Lindenwood',
        'stacked': 1,
        'annoline1': 'Lindenwood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.84963782402441,
         40.66391841925139,
         -73.84963782402441,
         40.66391841925139]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.189',
       'geometry': {'type': 'Point',
        'coordinates': [-73.74025607989822, 40.66788389660247]},
       'geometry_name': 'geom',
       'properties': {'name': 'Laurelton',
        'stacked': 1,
        'annoline1': 'Laurelton',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.74025607989822,
         40.66788389660247,
         -73.74025607989822,
         40.66788389660247]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.190',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8625247141374, 40.736074570830795]},
       'geometry_name': 'geom',
       'properties': {'name': 'Lefrak City',
        'stacked': 2,
        'annoline1': 'Lefrak',
        'annoline2': 'City',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.8625247141374,
         40.736074570830795,
         -73.8625247141374,
         40.736074570830795]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.191',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8540175039252, 40.57615556543109]},
       'geometry_name': 'geom',
       'properties': {'name': 'Belle Harbor',
        'stacked': 1,
        'annoline1': 'Belle Harbor',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.8540175039252,
         40.57615556543109,
         -73.8540175039252,
         40.57615556543109]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.192',
       'geometry': {'type': 'Point',
        'coordinates': [-73.84153370226186, 40.58034295646131]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rockaway Park',
        'stacked': 1,
        'annoline1': 'Rockaway Park',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.84153370226186,
         40.58034295646131,
         -73.84153370226186,
         40.58034295646131]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.193',
       'geometry': {'type': 'Point',
        'coordinates': [-73.79664750844047, 40.59771061565768]},
       'geometry_name': 'geom',
       'properties': {'name': 'Somerville',
        'stacked': 1,
        'annoline1': 'Somerville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.79664750844047,
         40.59771061565768,
         -73.79664750844047,
         40.59771061565768]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.194',
       'geometry': {'type': 'Point',
        'coordinates': [-73.75175310731153, 40.66000322733613]},
       'geometry_name': 'geom',
       'properties': {'name': 'Brookville',
        'stacked': 1,
        'annoline1': 'Brookville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.75175310731153,
         40.66000322733613,
         -73.75175310731153,
         40.66000322733613]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.195',
       'geometry': {'type': 'Point',
        'coordinates': [-73.73889198912481, 40.73301404027834]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bellaire',
        'stacked': 1,
        'annoline1': 'Bellaire',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.73889198912481,
         40.73301404027834,
         -73.73889198912481,
         40.73301404027834]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.196',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85751790676447, 40.7540709990489]},
       'geometry_name': 'geom',
       'properties': {'name': 'North Corona',
        'stacked': 2,
        'annoline1': 'North',
        'annoline2': 'Corona',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.85751790676447,
         40.7540709990489,
         -73.85751790676447,
         40.7540709990489]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.197',
       'geometry': {'type': 'Point',
        'coordinates': [-73.8410221123401, 40.7146110815117]},
       'geometry_name': 'geom',
       'properties': {'name': 'Forest Hills Gardens',
        'stacked': 3,
        'annoline1': 'Forest',
        'annoline2': 'Hills',
        'annoline3': 'Gardens',
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.8410221123401,
         40.7146110815117,
         -73.8410221123401,
         40.7146110815117]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.198',
       'geometry': {'type': 'Point',
        'coordinates': [-74.07935312512797, 40.6449815710044]},
       'geometry_name': 'geom',
       'properties': {'name': 'St. George',
        'stacked': 2,
        'annoline1': 'St.',
        'annoline2': 'George',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.07935312512797,
         40.6449815710044,
         -74.07935312512797,
         40.6449815710044]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.199',
       'geometry': {'type': 'Point',
        'coordinates': [-74.08701650516625, 40.64061455913511]},
       'geometry_name': 'geom',
       'properties': {'name': 'New Brighton',
        'stacked': 2,
        'annoline1': 'New',
        'annoline2': 'Brighton',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.08701650516625,
         40.64061455913511,
         -74.08701650516625,
         40.64061455913511]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.200',
       'geometry': {'type': 'Point',
        'coordinates': [-74.07790192660066, 40.62692762538176]},
       'geometry_name': 'geom',
       'properties': {'name': 'Stapleton',
        'stacked': 1,
        'annoline1': 'Stapleton',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.07790192660066,
         40.62692762538176,
         -74.07790192660066,
         40.62692762538176]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.201',
       'geometry': {'type': 'Point',
        'coordinates': [-74.06980526716141, 40.61530494652761]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rosebank',
        'stacked': 1,
        'annoline1': 'Rosebank',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.06980526716141,
         40.61530494652761,
         -74.06980526716141,
         40.61530494652761]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.202',
       'geometry': {'type': 'Point',
        'coordinates': [-74.1071817826561, 40.63187892654607]},
       'geometry_name': 'geom',
       'properties': {'name': 'West Brighton',
        'stacked': 2,
        'annoline1': 'West',
        'annoline2': 'Brighton',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.1071817826561,
         40.63187892654607,
         -74.1071817826561,
         40.63187892654607]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.203',
       'geometry': {'type': 'Point',
        'coordinates': [-74.08724819983729, 40.624184791313006]},
       'geometry_name': 'geom',
       'properties': {'name': 'Grymes Hill',
        'stacked': 2,
        'annoline1': 'Grymes',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.08724819983729,
         40.624184791313006,
         -74.08724819983729,
         40.624184791313006]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.204',
       'geometry': {'type': 'Point',
        'coordinates': [-74.1113288180088, 40.59706851814673]},
       'geometry_name': 'geom',
       'properties': {'name': 'Todt Hill',
        'stacked': 2,
        'annoline1': 'Todt',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.1113288180088,
         40.59706851814673,
         -74.1113288180088,
         40.59706851814673]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.205',
       'geometry': {'type': 'Point',
        'coordinates': [-74.0795529253982, 40.58024741350956]},
       'geometry_name': 'geom',
       'properties': {'name': 'South Beach',
        'stacked': 2,
        'annoline1': 'South',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.0795529253982,
         40.58024741350956,
         -74.0795529253982,
         40.58024741350956]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.206',
       'geometry': {'type': 'Point',
        'coordinates': [-74.12943426797008, 40.63366930554365]},
       'geometry_name': 'geom',
       'properties': {'name': 'Port Richmond',
        'stacked': 2,
        'annoline1': 'Port',
        'annoline2': 'Richmond',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.12943426797008,
         40.63366930554365,
         -74.12943426797008,
         40.63366930554365]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.207',
       'geometry': {'type': 'Point',
        'coordinates': [-74.15008537046981, 40.632546390481124]},
       'geometry_name': 'geom',
       'properties': {'name': "Mariner's Harbor",
        'stacked': 2,
        'annoline1': "Mariner's",
        'annoline2': 'Harbor',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.15008537046981,
         40.632546390481124,
         -74.15008537046981,
         40.632546390481124]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.208',
       'geometry': {'type': 'Point',
        'coordinates': [-74.17464532993542, 40.63968297845542]},
       'geometry_name': 'geom',
       'properties': {'name': 'Port Ivory',
        'stacked': 2,
        'annoline1': 'Port',
        'annoline2': 'Ivory',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.17464532993542,
         40.63968297845542,
         -74.17464532993542,
         40.63968297845542]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.209',
       'geometry': {'type': 'Point',
        'coordinates': [-74.11918058534842, 40.61333593766742]},
       'geometry_name': 'geom',
       'properties': {'name': 'Castleton Corners',
        'stacked': 2,
        'annoline1': 'Castleton',
        'annoline2': 'Corners',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.11918058534842,
         40.61333593766742,
         -74.11918058534842,
         40.61333593766742]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.210',
       'geometry': {'type': 'Point',
        'coordinates': [-74.16496031329827, 40.594252379161695]},
       'geometry_name': 'geom',
       'properties': {'name': 'New Springville',
        'stacked': 2,
        'annoline1': 'New',
        'annoline2': 'Springville',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.16496031329827,
         40.594252379161695,
         -74.16496031329827,
         40.594252379161695]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.211',
       'geometry': {'type': 'Point',
        'coordinates': [-74.19073717538116, 40.58631375103281]},
       'geometry_name': 'geom',
       'properties': {'name': 'Travis',
        'stacked': 1,
        'annoline1': 'Travis',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.19073717538116,
         40.58631375103281,
         -74.19073717538116,
         40.58631375103281]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.212',
       'geometry': {'type': 'Point',
        'coordinates': [-74.1164794360638, 40.57257231820632]},
       'geometry_name': 'geom',
       'properties': {'name': 'New Dorp',
        'stacked': 2,
        'annoline1': 'New',
        'annoline2': 'Dorp',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.1164794360638,
         40.57257231820632,
         -74.1164794360638,
         40.57257231820632]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.213',
       'geometry': {'type': 'Point',
        'coordinates': [-74.12156593771896, 40.5584622432888]},
       'geometry_name': 'geom',
       'properties': {'name': 'Oakwood',
        'stacked': 1,
        'annoline1': 'Oakwood',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.12156593771896,
         40.5584622432888,
         -74.12156593771896,
         40.5584622432888]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.214',
       'geometry': {'type': 'Point',
        'coordinates': [-74.14932381490992, 40.549480228713605]},
       'geometry_name': 'geom',
       'properties': {'name': 'Great Kills',
        'stacked': 2,
        'annoline1': 'Great',
        'annoline2': 'Kills',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.14932381490992,
         40.549480228713605,
         -74.14932381490992,
         40.549480228713605]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.215',
       'geometry': {'type': 'Point',
        'coordinates': [-74.1643308041936, 40.542230747450745]},
       'geometry_name': 'geom',
       'properties': {'name': 'Eltingville',
        'stacked': 1,
        'annoline1': 'Eltingville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.1643308041936,
         40.542230747450745,
         -74.1643308041936,
         40.542230747450745]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.216',
       'geometry': {'type': 'Point',
        'coordinates': [-74.17854866165878, 40.53811417474507]},
       'geometry_name': 'geom',
       'properties': {'name': 'Annadale',
        'stacked': 1,
        'annoline1': 'Annadale',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.17854866165878,
         40.53811417474507,
         -74.17854866165878,
         40.53811417474507]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.217',
       'geometry': {'type': 'Point',
        'coordinates': [-74.20524582480326, 40.541967622888755]},
       'geometry_name': 'geom',
       'properties': {'name': 'Woodrow',
        'stacked': 1,
        'annoline1': 'Woodrow',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.20524582480326,
         40.541967622888755,
         -74.20524582480326,
         40.541967622888755]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.218',
       'geometry': {'type': 'Point',
        'coordinates': [-74.24656934235283, 40.50533376115642]},
       'geometry_name': 'geom',
       'properties': {'name': 'Tottenville',
        'stacked': 1,
        'annoline1': 'Tottenville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.24656934235283,
         40.50533376115642,
         -74.24656934235283,
         40.50533376115642]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.219',
       'geometry': {'type': 'Point',
        'coordinates': [-74.08055351790115, 40.637316067110326]},
       'geometry_name': 'geom',
       'properties': {'name': 'Tompkinsville',
        'stacked': 1,
        'annoline1': 'Tompkinsville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.08055351790115,
         40.637316067110326,
         -74.08055351790115,
         40.637316067110326]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.220',
       'geometry': {'type': 'Point',
        'coordinates': [-74.09629029235458, 40.61919310792676]},
       'geometry_name': 'geom',
       'properties': {'name': 'Silver Lake',
        'stacked': 2,
        'annoline1': 'Silver',
        'annoline2': 'Lake',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.09629029235458,
         40.61919310792676,
         -74.09629029235458,
         40.61919310792676]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.221',
       'geometry': {'type': 'Point',
        'coordinates': [-74.0971255217853, 40.61276015756489]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sunnyside',
        'stacked': 1,
        'annoline1': 'Sunnyside',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.0971255217853,
         40.61276015756489,
         -74.0971255217853,
         40.61276015756489]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.222',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96101312466779, 40.643675183340974]},
       'geometry_name': 'geom',
       'properties': {'name': 'Ditmas Park',
        'stacked': 2,
        'annoline1': 'Ditmas',
        'annoline2': 'Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.96101312466779,
         40.643675183340974,
         -73.96101312466779,
         40.643675183340974]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.223',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93718680559314, 40.66094656188111]},
       'geometry_name': 'geom',
       'properties': {'name': 'Wingate',
        'stacked': 1,
        'annoline1': 'Wingate',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.93718680559314,
         40.66094656188111,
         -73.93718680559314,
         40.66094656188111]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.224',
       'geometry': {'type': 'Point',
        'coordinates': [-73.92688212616955, 40.655572313280764]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rugby',
        'stacked': 1,
        'annoline1': 'Rugby',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.92688212616955,
         40.655572313280764,
         -73.92688212616955,
         40.655572313280764]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.225',
       'geometry': {'type': 'Point',
        'coordinates': [-74.08015734936296, 40.60919044434558]},
       'geometry_name': 'geom',
       'properties': {'name': 'Park Hill',
        'stacked': 2,
        'annoline1': 'Park',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.08015734936296,
         40.60919044434558,
         -74.08015734936296,
         40.60919044434558]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.226',
       'geometry': {'type': 'Point',
        'coordinates': [-74.13304143951704, 40.62109047275409]},
       'geometry_name': 'geom',
       'properties': {'name': 'Westerleigh',
        'stacked': 1,
        'annoline1': 'Westerleigh',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.13304143951704,
         40.62109047275409,
         -74.13304143951704,
         40.62109047275409]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.227',
       'geometry': {'type': 'Point',
        'coordinates': [-74.15315246387762, 40.620171512231884]},
       'geometry_name': 'geom',
       'properties': {'name': 'Graniteville',
        'stacked': 1,
        'annoline1': 'Graniteville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.15315246387762,
         40.620171512231884,
         -74.15315246387762,
         40.620171512231884]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.228',
       'geometry': {'type': 'Point',
        'coordinates': [-74.16510420241124, 40.63532509911492]},
       'geometry_name': 'geom',
       'properties': {'name': 'Arlington',
        'stacked': 1,
        'annoline1': 'Arlington',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.16510420241124,
         40.63532509911492,
         -74.16510420241124,
         40.63532509911492]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.229',
       'geometry': {'type': 'Point',
        'coordinates': [-74.06712363225574, 40.596312571276734]},
       'geometry_name': 'geom',
       'properties': {'name': 'Arrochar',
        'stacked': 1,
        'annoline1': 'Arrochar',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.06712363225574,
         40.596312571276734,
         -74.06712363225574,
         40.596312571276734]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.230',
       'geometry': {'type': 'Point',
        'coordinates': [-74.0766743627905, 40.59826835959991]},
       'geometry_name': 'geom',
       'properties': {'name': 'Grasmere',
        'stacked': 1,
        'annoline1': 'Grasmere',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.0766743627905,
         40.59826835959991,
         -74.0766743627905,
         40.59826835959991]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.231',
       'geometry': {'type': 'Point',
        'coordinates': [-74.08751118005578, 40.59632891379513]},
       'geometry_name': 'geom',
       'properties': {'name': 'Old Town',
        'stacked': 2,
        'annoline1': 'Old',
        'annoline2': 'Town',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.08751118005578,
         40.59632891379513,
         -74.08751118005578,
         40.59632891379513]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.232',
       'geometry': {'type': 'Point',
        'coordinates': [-74.09639905312521, 40.588672948199275]},
       'geometry_name': 'geom',
       'properties': {'name': 'Dongan Hills',
        'stacked': 2,
        'annoline1': 'Dongan',
        'annoline2': 'Hills',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.09639905312521,
         40.588672948199275,
         -74.09639905312521,
         40.588672948199275]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.233',
       'geometry': {'type': 'Point',
        'coordinates': [-74.09348266303591, 40.57352690574283]},
       'geometry_name': 'geom',
       'properties': {'name': 'Midland Beach',
        'stacked': 2,
        'annoline1': 'Midland',
        'annoline2': 'Beach',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.09348266303591,
         40.57352690574283,
         -74.09348266303591,
         40.57352690574283]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.234',
       'geometry': {'type': 'Point',
        'coordinates': [-74.10585598545434, 40.57621558711788]},
       'geometry_name': 'geom',
       'properties': {'name': 'Grant City',
        'stacked': 2,
        'annoline1': 'Grant',
        'annoline2': 'City',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.10585598545434,
         40.57621558711788,
         -74.10585598545434,
         40.57621558711788]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.235',
       'geometry': {'type': 'Point',
        'coordinates': [-74.10432707469124, 40.56425549307335]},
       'geometry_name': 'geom',
       'properties': {'name': 'New Dorp Beach',
        'stacked': 3,
        'annoline1': 'New',
        'annoline2': 'Dorp',
        'annoline3': 'Beach',
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.10432707469124,
         40.56425549307335,
         -74.10432707469124,
         40.56425549307335]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.236',
       'geometry': {'type': 'Point',
        'coordinates': [-74.13916622175768, 40.55398800858462]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bay Terrace',
        'stacked': 2,
        'annoline1': 'Bay',
        'annoline2': 'Terrace',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.13916622175768,
         40.55398800858462,
         -74.13916622175768,
         40.55398800858462]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.237',
       'geometry': {'type': 'Point',
        'coordinates': [-74.19174105747814, 40.531911920489605]},
       'geometry_name': 'geom',
       'properties': {'name': 'Huguenot',
        'stacked': 1,
        'annoline1': 'Huguenot',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.19174105747814,
         40.531911920489605,
         -74.19174105747814,
         40.531911920489605]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.238',
       'geometry': {'type': 'Point',
        'coordinates': [-74.21983106616777, 40.524699376118136]},
       'geometry_name': 'geom',
       'properties': {'name': 'Pleasant Plains',
        'stacked': 2,
        'annoline1': 'Pleasant',
        'annoline2': 'Plains',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.21983106616777,
         40.524699376118136,
         -74.21983106616777,
         40.524699376118136]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.239',
       'geometry': {'type': 'Point',
        'coordinates': [-74.22950350260027, 40.50608165346305]},
       'geometry_name': 'geom',
       'properties': {'name': 'Butler Manor',
        'stacked': 2,
        'annoline1': 'Butler',
        'annoline2': 'Manor',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.22950350260027,
         40.50608165346305,
         -74.22950350260027,
         40.50608165346305]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.240',
       'geometry': {'type': 'Point',
        'coordinates': [-74.23215775896526, 40.53053148283314]},
       'geometry_name': 'geom',
       'properties': {'name': 'Charleston',
        'stacked': 1,
        'annoline1': 'Charleston',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.23215775896526,
         40.53053148283314,
         -74.23215775896526,
         40.53053148283314]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.241',
       'geometry': {'type': 'Point',
        'coordinates': [-74.21572851113952, 40.54940400650072]},
       'geometry_name': 'geom',
       'properties': {'name': 'Rossville',
        'stacked': 1,
        'annoline1': 'Rossville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.21572851113952,
         40.54940400650072,
         -74.21572851113952,
         40.54940400650072]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.242',
       'geometry': {'type': 'Point',
        'coordinates': [-74.18588674583893, 40.54928582278321]},
       'geometry_name': 'geom',
       'properties': {'name': 'Arden Heights',
        'stacked': 2,
        'annoline1': 'Arden',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.18588674583893,
         40.54928582278321,
         -74.18588674583893,
         40.54928582278321]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.243',
       'geometry': {'type': 'Point',
        'coordinates': [-74.17079414786092, 40.555295236173194]},
       'geometry_name': 'geom',
       'properties': {'name': 'Greenridge',
        'stacked': 1,
        'annoline1': 'Greenridge',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.17079414786092,
         40.555295236173194,
         -74.17079414786092,
         40.555295236173194]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.244',
       'geometry': {'type': 'Point',
        'coordinates': [-74.15902208156601, 40.58913894875281]},
       'geometry_name': 'geom',
       'properties': {'name': 'Heartland Village',
        'stacked': 2,
        'annoline1': 'Heartland',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.15902208156601,
         40.58913894875281,
         -74.15902208156601,
         40.58913894875281]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.245',
       'geometry': {'type': 'Point',
        'coordinates': [-74.1895604551969, 40.59472602746295]},
       'geometry_name': 'geom',
       'properties': {'name': 'Chelsea',
        'stacked': 1,
        'annoline1': 'Chelsea',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.1895604551969,
         40.59472602746295,
         -74.1895604551969,
         40.59472602746295]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.246',
       'geometry': {'type': 'Point',
        'coordinates': [-74.18725638381567, 40.60577868452358]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bloomfield',
        'stacked': 1,
        'annoline1': 'Bloomfield',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.18725638381567,
         40.60577868452358,
         -74.18725638381567,
         40.60577868452358]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.247',
       'geometry': {'type': 'Point',
        'coordinates': [-74.15940948657122, 40.6095918004203]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bulls Head',
        'stacked': 2,
        'annoline1': 'Bulls',
        'annoline2': 'Head',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.15940948657122,
         40.6095918004203,
         -74.15940948657122,
         40.6095918004203]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.248',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95325646837112, 40.7826825671257]},
       'geometry_name': 'geom',
       'properties': {'name': 'Carnegie Hill',
        'stacked': 2,
        'annoline1': 'Carnegie',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.95325646837112,
         40.7826825671257,
         -73.95325646837112,
         40.7826825671257]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.249',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98843368023597, 40.72325901885768]},
       'geometry_name': 'geom',
       'properties': {'name': 'Noho',
        'stacked': 1,
        'annoline1': 'Noho',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.98843368023597,
         40.72325901885768,
         -73.98843368023597,
         40.72325901885768]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.250',
       'geometry': {'type': 'Point',
        'coordinates': [-74.00541529873355, 40.71522892046282]},
       'geometry_name': 'geom',
       'properties': {'name': 'Civic Center',
        'stacked': 2,
        'annoline1': 'Civic',
        'annoline2': 'Center',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.00541529873355,
         40.71522892046282,
         -74.00541529873355,
         40.71522892046282]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.251',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98871313285247, 40.7485096643122]},
       'geometry_name': 'geom',
       'properties': {'name': 'Midtown South',
        'stacked': 2,
        'annoline1': 'Midtown',
        'annoline2': 'South',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.98871313285247,
         40.7485096643122,
         -73.98871313285247,
         40.7485096643122]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.252',
       'geometry': {'type': 'Point',
        'coordinates': [-74.1340572986257, 40.56960594275505]},
       'geometry_name': 'geom',
       'properties': {'name': 'Richmond Town',
        'stacked': 2,
        'annoline1': 'Richmond',
        'annoline2': 'Town',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.1340572986257,
         40.56960594275505,
         -74.1340572986257,
         40.56960594275505]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.253',
       'geometry': {'type': 'Point',
        'coordinates': [-74.06667766061771, 40.60971934079284]},
       'geometry_name': 'geom',
       'properties': {'name': 'Shore Acres',
        'stacked': 2,
        'annoline1': 'Shore',
        'annoline2': 'Acres',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.06667766061771,
         40.60971934079284,
         -74.06667766061771,
         40.60971934079284]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.254',
       'geometry': {'type': 'Point',
        'coordinates': [-74.072642445484, 40.61917845202843]},
       'geometry_name': 'geom',
       'properties': {'name': 'Clifton',
        'stacked': 1,
        'annoline1': 'Clifton',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.072642445484,
         40.61917845202843,
         -74.072642445484,
         40.61917845202843]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.255',
       'geometry': {'type': 'Point',
        'coordinates': [-74.08402364740358, 40.6044731896879]},
       'geometry_name': 'geom',
       'properties': {'name': 'Concord',
        'stacked': 1,
        'annoline1': 'Concord',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.08402364740358,
         40.6044731896879,
         -74.08402364740358,
         40.6044731896879]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.256',
       'geometry': {'type': 'Point',
        'coordinates': [-74.09776206972522, 40.606794394801]},
       'geometry_name': 'geom',
       'properties': {'name': 'Emerson Hill',
        'stacked': 2,
        'annoline1': 'Emerson',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.09776206972522,
         40.606794394801,
         -74.09776206972522,
         40.606794394801]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.257',
       'geometry': {'type': 'Point',
        'coordinates': [-74.09805062373887, 40.63563000681151]},
       'geometry_name': 'geom',
       'properties': {'name': 'Randall Manor',
        'stacked': 2,
        'annoline1': 'Randall',
        'annoline2': 'Manor',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.09805062373887,
         40.63563000681151,
         -74.09805062373887,
         40.63563000681151]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.258',
       'geometry': {'type': 'Point',
        'coordinates': [-74.18622331749823, 40.63843283794795]},
       'geometry_name': 'geom',
       'properties': {'name': 'Howland Hook',
        'stacked': 2,
        'annoline1': 'Howland',
        'annoline2': 'Hook',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.18622331749823,
         40.63843283794795,
         -74.18622331749823,
         40.63843283794795]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.259',
       'geometry': {'type': 'Point',
        'coordinates': [-74.1418167896889, 40.630146741193826]},
       'geometry_name': 'geom',
       'properties': {'name': 'Elm Park',
        'stacked': 2,
        'annoline1': 'Elm',
        'annoline2': 'Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.1418167896889,
         40.630146741193826,
         -74.1418167896889,
         40.630146741193826]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.260',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91665331978048, 40.652117451793494]},
       'geometry_name': 'geom',
       'properties': {'name': 'Remsen Village',
        'stacked': 2,
        'annoline1': 'Remsen',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.91665331978048,
         40.652117451793494,
         -73.91665331978048,
         40.652117451793494]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.261',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88511776379292, 40.6627442796966]},
       'geometry_name': 'geom',
       'properties': {'name': 'New Lots',
        'stacked': 2,
        'annoline1': 'New',
        'annoline2': 'Lots',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.88511776379292,
         40.6627442796966,
         -73.88511776379292,
         40.6627442796966]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.262',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90233474295836, 40.63131755039667]},
       'geometry_name': 'geom',
       'properties': {'name': 'Paerdegat Basin',
        'stacked': 2,
        'annoline1': 'Paerdegat',
        'annoline2': 'Basin',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.90233474295836,
         40.63131755039667,
         -73.90233474295836,
         40.63131755039667]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.263',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91515391550404, 40.61597423962336]},
       'geometry_name': 'geom',
       'properties': {'name': 'Mill Basin',
        'stacked': 2,
        'annoline1': 'Mill',
        'annoline2': 'Basin',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.91515391550404,
         40.61597423962336,
         -73.91515391550404,
         40.61597423962336]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.264',
       'geometry': {'type': 'Point',
        'coordinates': [-73.79646462081593, 40.71145964370482]},
       'geometry_name': 'geom',
       'properties': {'name': 'Jamaica Hills',
        'stacked': 2,
        'annoline1': 'Jamaica',
        'annoline2': 'Hills',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.79646462081593,
         40.71145964370482,
         -73.79646462081593,
         40.71145964370482]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.265',
       'geometry': {'type': 'Point',
        'coordinates': [-73.79671678028349, 40.73350025429757]},
       'geometry_name': 'geom',
       'properties': {'name': 'Utopia',
        'stacked': 1,
        'annoline1': 'Utopia',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.79671678028349,
         40.73350025429757,
         -73.79671678028349,
         40.73350025429757]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.266',
       'geometry': {'type': 'Point',
        'coordinates': [-73.80486120040537, 40.73493618075478]},
       'geometry_name': 'geom',
       'properties': {'name': 'Pomonok',
        'stacked': 1,
        'annoline1': 'Pomonok',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.80486120040537,
         40.73493618075478,
         -73.80486120040537,
         40.73493618075478]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.267',
       'geometry': {'type': 'Point',
        'coordinates': [-73.89467996270574, 40.7703173929982]},
       'geometry_name': 'geom',
       'properties': {'name': 'Astoria Heights',
        'stacked': 2,
        'annoline1': 'Astoria',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.89467996270574,
         40.7703173929982,
         -73.89467996270574,
         40.7703173929982]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.268',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90119903387667, 40.83142834161548]},
       'geometry_name': 'geom',
       'properties': {'name': 'Claremont Village',
        'stacked': 2,
        'annoline1': 'Claremont',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90119903387667,
         40.83142834161548,
         -73.90119903387667,
         40.83142834161548]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.269',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91584652759009, 40.824780490842905]},
       'geometry_name': 'geom',
       'properties': {'name': 'Concourse Village',
        'stacked': 2,
        'annoline1': 'Concourse',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.91584652759009,
         40.824780490842905,
         -73.91584652759009,
         40.824780490842905]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.270',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91655551964419, 40.84382617671654]},
       'geometry_name': 'geom',
       'properties': {'name': 'Mount Eden',
        'stacked': 2,
        'annoline1': 'Mount',
        'annoline2': 'Eden',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.91655551964419,
         40.84382617671654,
         -73.91655551964419,
         40.84382617671654]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.271',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90829930881988, 40.84884160724665]},
       'geometry_name': 'geom',
       'properties': {'name': 'Mount Hope',
        'stacked': 2,
        'annoline1': 'Mount',
        'annoline2': 'Hope',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90829930881988,
         40.84884160724665,
         -73.90829930881988,
         40.84884160724665]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.272',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96355614094303, 40.76028033131374]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sutton Place',
        'stacked': 2,
        'annoline1': 'Sutton',
        'annoline2': 'Place',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.96355614094303,
         40.76028033131374,
         -73.96355614094303,
         40.76028033131374]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.273',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95386782130745, 40.743414090073536]},
       'geometry_name': 'geom',
       'properties': {'name': 'Hunters Point',
        'stacked': 2,
        'annoline1': 'Hunters',
        'annoline2': 'Point',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.95386782130745,
         40.743414090073536,
         -73.95386782130745,
         40.743414090073536]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.274',
       'geometry': {'type': 'Point',
        'coordinates': [-73.96770824581834, 40.75204236950722]},
       'geometry_name': 'geom',
       'properties': {'name': 'Turtle Bay',
        'stacked': 2,
        'annoline1': 'Turtle',
        'annoline2': 'Bay',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.96770824581834,
         40.75204236950722,
         -73.96770824581834,
         40.75204236950722]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.275',
       'geometry': {'type': 'Point',
        'coordinates': [-73.97121928722265, 40.746917410740195]},
       'geometry_name': 'geom',
       'properties': {'name': 'Tudor City',
        'stacked': 2,
        'annoline1': 'Tudor',
        'annoline2': 'City',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.97121928722265,
         40.746917410740195,
         -73.97121928722265,
         40.746917410740195]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.276',
       'geometry': {'type': 'Point',
        'coordinates': [-73.97405170469203, 40.73099955477061]},
       'geometry_name': 'geom',
       'properties': {'name': 'Stuyvesant Town',
        'stacked': 2,
        'annoline1': 'Stuyvesant',
        'annoline2': 'Town',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.97405170469203,
         40.73099955477061,
         -73.97405170469203,
         40.73099955477061]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.277',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9909471052826, 40.739673047638426]},
       'geometry_name': 'geom',
       'properties': {'name': 'Flatiron',
        'stacked': 1,
        'annoline1': 'Flatiron',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-73.9909471052826,
         40.739673047638426,
         -73.9909471052826,
         40.739673047638426]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.278',
       'geometry': {'type': 'Point',
        'coordinates': [-73.91819286431682, 40.74565180608076]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sunnyside Gardens',
        'stacked': 2,
        'annoline1': 'Sunnyside',
        'annoline2': 'Gardens',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.91819286431682,
         40.74565180608076,
         -73.91819286431682,
         40.74565180608076]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.279',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93244235260178, 40.73725071694497]},
       'geometry_name': 'geom',
       'properties': {'name': 'Blissville',
        'stacked': 1,
        'annoline1': 'Blissville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.93244235260178,
         40.73725071694497,
         -73.93244235260178,
         40.73725071694497]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.280',
       'geometry': {'type': 'Point',
        'coordinates': [-73.99550751888415, 40.70328109093014]},
       'geometry_name': 'geom',
       'properties': {'name': 'Fulton Ferry',
        'stacked': 2,
        'annoline1': 'Fulton',
        'annoline2': 'Ferry',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.99550751888415,
         40.70328109093014,
         -73.99550751888415,
         40.70328109093014]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.281',
       'geometry': {'type': 'Point',
        'coordinates': [-73.98111603592393, 40.70332149882874]},
       'geometry_name': 'geom',
       'properties': {'name': 'Vinegar Hill',
        'stacked': 2,
        'annoline1': 'Vinegar',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.98111603592393,
         40.70332149882874,
         -73.98111603592393,
         40.70332149882874]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.282',
       'geometry': {'type': 'Point',
        'coordinates': [-73.93053108817338, 40.67503986503237]},
       'geometry_name': 'geom',
       'properties': {'name': 'Weeksville',
        'stacked': 1,
        'annoline1': 'Weeksville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.93053108817338,
         40.67503986503237,
         -73.93053108817338,
         40.67503986503237]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.283',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90331684852599, 40.67786104769531]},
       'geometry_name': 'geom',
       'properties': {'name': 'Broadway Junction',
        'stacked': 2,
        'annoline1': 'Broadway',
        'annoline2': 'Junction',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.90331684852599,
         40.67786104769531,
         -73.90331684852599,
         40.67786104769531]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.284',
       'geometry': {'type': 'Point',
        'coordinates': [-73.9887528074504, 40.70317632822692]},
       'geometry_name': 'geom',
       'properties': {'name': 'Dumbo',
        'stacked': 1,
        'annoline1': 'Dumbo',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.9887528074504,
         40.70317632822692,
         -73.9887528074504,
         40.70317632822692]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.285',
       'geometry': {'type': 'Point',
        'coordinates': [-74.12059399718001, 40.60180957631444]},
       'geometry_name': 'geom',
       'properties': {'name': 'Manor Heights',
        'stacked': 2,
        'annoline1': 'Manor',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.12059399718001,
         40.60180957631444,
         -74.12059399718001,
         40.60180957631444]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.286',
       'geometry': {'type': 'Point',
        'coordinates': [-74.13208447484298, 40.60370692627371]},
       'geometry_name': 'geom',
       'properties': {'name': 'Willowbrook',
        'stacked': 1,
        'annoline1': 'Willowbrook',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.13208447484298,
         40.60370692627371,
         -74.13208447484298,
         40.60370692627371]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.287',
       'geometry': {'type': 'Point',
        'coordinates': [-74.21776636068567, 40.541139922091766]},
       'geometry_name': 'geom',
       'properties': {'name': 'Sandy Ground',
        'stacked': 2,
        'annoline1': 'Sandy',
        'annoline2': 'Ground',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.21776636068567,
         40.541139922091766,
         -74.21776636068567,
         40.541139922091766]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.288',
       'geometry': {'type': 'Point',
        'coordinates': [-74.12727240604946, 40.579118742961214]},
       'geometry_name': 'geom',
       'properties': {'name': 'Egbertville',
        'stacked': 1,
        'annoline1': 'Egbertville',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.12727240604946,
         40.579118742961214,
         -74.12727240604946,
         40.579118742961214]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.289',
       'geometry': {'type': 'Point',
        'coordinates': [-73.89213760232822, 40.56737588957032]},
       'geometry_name': 'geom',
       'properties': {'name': 'Roxbury',
        'stacked': 1,
        'annoline1': 'Roxbury',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.89213760232822,
         40.56737588957032,
         -73.89213760232822,
         40.56737588957032]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.290',
       'geometry': {'type': 'Point',
        'coordinates': [-73.95918459428702, 40.598525095137255]},
       'geometry_name': 'geom',
       'properties': {'name': 'Homecrest',
        'stacked': 1,
        'annoline1': 'Homecrest',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.95918459428702,
         40.598525095137255,
         -73.95918459428702,
         40.598525095137255]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.291',
       'geometry': {'type': 'Point',
        'coordinates': [-73.88114319200604, 40.716414511158185]},
       'geometry_name': 'geom',
       'properties': {'name': 'Middle Village',
        'stacked': 2,
        'annoline1': 'Middle',
        'annoline2': 'Village',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.88114319200604,
         40.716414511158185,
         -73.88114319200604,
         40.716414511158185]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.292',
       'geometry': {'type': 'Point',
        'coordinates': [-74.20152556457658, 40.52626406734812]},
       'geometry_name': 'geom',
       'properties': {'name': "Prince's Bay",
        'stacked': 2,
        'annoline1': "Prince's",
        'annoline2': 'Bay',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.20152556457658,
         40.52626406734812,
         -74.20152556457658,
         40.52626406734812]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.293',
       'geometry': {'type': 'Point',
        'coordinates': [-74.13792663771568, 40.57650629379489]},
       'geometry_name': 'geom',
       'properties': {'name': 'Lighthouse Hill',
        'stacked': 2,
        'annoline1': 'Lighthouse',
        'annoline2': 'Hill',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.13792663771568,
         40.57650629379489,
         -74.13792663771568,
         40.57650629379489]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.294',
       'geometry': {'type': 'Point',
        'coordinates': [-74.22957080626941, 40.51954145748909]},
       'geometry_name': 'geom',
       'properties': {'name': 'Richmond Valley',
        'stacked': 2,
        'annoline1': 'Richmond',
        'annoline2': 'Valley',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.22957080626941,
         40.51954145748909,
         -74.22957080626941,
         40.51954145748909]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.295',
       'geometry': {'type': 'Point',
        'coordinates': [-73.82667757138641, 40.79060155670148]},
       'geometry_name': 'geom',
       'properties': {'name': 'Malba',
        'stacked': 1,
        'annoline1': 'Malba',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.82667757138641,
         40.79060155670148,
         -73.82667757138641,
         40.79060155670148]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.296',
       'geometry': {'type': 'Point',
        'coordinates': [-73.890345709872, 40.6819989345173]},
       'geometry_name': 'geom',
       'properties': {'name': 'Highland Park',
        'stacked': 2,
        'annoline1': 'Highland',
        'annoline2': 'Park',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.890345709872,
         40.6819989345173,
         -73.890345709872,
         40.6819989345173]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.297',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94841515328893, 40.60937770113766]},
       'geometry_name': 'geom',
       'properties': {'name': 'Madison',
        'stacked': 1,
        'annoline1': 'Madison',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.94841515328893,
         40.60937770113766,
         -73.94841515328893,
         40.60937770113766]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.298',
       'geometry': {'type': 'Point',
        'coordinates': [-73.86172577555115, 40.85272297633017]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bronxdale',
        'stacked': 1,
        'annoline1': 'Bronxdale',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.86172577555115,
         40.85272297633017,
         -73.86172577555115,
         40.85272297633017]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.299',
       'geometry': {'type': 'Point',
        'coordinates': [-73.85931863221647, 40.86578787802982]},
       'geometry_name': 'geom',
       'properties': {'name': 'Allerton',
        'stacked': 1,
        'annoline1': 'Allerton',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.85931863221647,
         40.86578787802982,
         -73.85931863221647,
         40.86578787802982]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.300',
       'geometry': {'type': 'Point',
        'coordinates': [-73.90152264513144, 40.8703923914147]},
       'geometry_name': 'geom',
       'properties': {'name': 'Kingsbridge Heights',
        'stacked': 2,
        'annoline1': 'Kingsbridge',
        'annoline2': 'Heights',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Bronx',
        'bbox': [-73.90152264513144,
         40.8703923914147,
         -73.90152264513144,
         40.8703923914147]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.301',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94817709920184, 40.64692606658579]},
       'geometry_name': 'geom',
       'properties': {'name': 'Erasmus',
        'stacked': 1,
        'annoline1': 'Erasmus',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Brooklyn',
        'bbox': [-73.94817709920184,
         40.64692606658579,
         -73.94817709920184,
         40.64692606658579]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.302',
       'geometry': {'type': 'Point',
        'coordinates': [-74.00011136202637, 40.75665808227519]},
       'geometry_name': 'geom',
       'properties': {'name': 'Hudson Yards',
        'stacked': 2,
        'annoline1': 'Hudson',
        'annoline2': 'Yards',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Manhattan',
        'bbox': [-74.00011136202637,
         40.75665808227519,
         -74.00011136202637,
         40.75665808227519]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.303',
       'geometry': {'type': 'Point',
        'coordinates': [-73.80553002968718, 40.58733774018741]},
       'geometry_name': 'geom',
       'properties': {'name': 'Hammels',
        'stacked': 1,
        'annoline1': 'Hammels',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.80553002968718,
         40.58733774018741,
         -73.80553002968718,
         40.58733774018741]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.304',
       'geometry': {'type': 'Point',
        'coordinates': [-73.76596781445627, 40.611321691283834]},
       'geometry_name': 'geom',
       'properties': {'name': 'Bayswater',
        'stacked': 1,
        'annoline1': 'Bayswater',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.76596781445627,
         40.611321691283834,
         -73.76596781445627,
         40.611321691283834]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.305',
       'geometry': {'type': 'Point',
        'coordinates': [-73.94563070334091, 40.756091297094706]},
       'geometry_name': 'geom',
       'properties': {'name': 'Queensbridge',
        'stacked': 1,
        'annoline1': 'Queensbridge',
        'annoline2': None,
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Queens',
        'bbox': [-73.94563070334091,
         40.756091297094706,
         -73.94563070334091,
         40.756091297094706]}},
      {'type': 'Feature',
       'id': 'nyu_2451_34572.306',
       'geometry': {'type': 'Point',
        'coordinates': [-74.08173992211962, 40.61731079252983]},
       'geometry_name': 'geom',
       'properties': {'name': 'Fox Hills',
        'stacked': 2,
        'annoline1': 'Fox',
        'annoline2': 'Hills',
        'annoline3': None,
        'annoangle': 0.0,
        'borough': 'Staten Island',
        'bbox': [-74.08173992211962,
         40.61731079252983,
         -74.08173992211962,
         40.61731079252983]}}],
     'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::4326'}},
     'bbox': [-74.2492599487305,
      40.5033187866211,
      -73.7061614990234,
      40.9105606079102]}




```python
neighborhoods_data = newyork_data['features']
```


```python
neighborhoods_data[0]
```




    {'type': 'Feature',
     'id': 'nyu_2451_34572.1',
     'geometry': {'type': 'Point',
      'coordinates': [-73.84720052054902, 40.89470517661]},
     'geometry_name': 'geom',
     'properties': {'name': 'Wakefield',
      'stacked': 1,
      'annoline1': 'Wakefield',
      'annoline2': None,
      'annoline3': None,
      'annoangle': 0.0,
      'borough': 'Bronx',
      'bbox': [-73.84720052054902,
       40.89470517661,
       -73.84720052054902,
       40.89470517661]}}




```python
#Creating an empty dataframe for NYC
column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 

# instantiating the dataframe
neighborhoods = pd.DataFrame(columns=column_names)
```


```python
neighborhoods
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
for data in neighborhoods_data:
    borough = neighborhood_name = data['properties']['borough'] 
    neighborhood_name = data['properties']['name']
        
    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]
    
    neighborhoods = neighborhoods.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)
```


```python
neighborhoods.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>Co-op City</td>
      <td>40.874294</td>
      <td>-73.829939</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>Eastchester</td>
      <td>40.887556</td>
      <td>-73.827806</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>Fieldston</td>
      <td>40.895437</td>
      <td>-73.905643</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>Riverdale</td>
      <td>40.890834</td>
      <td>-73.912585</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(neighborhoods['Borough'].unique()),
        neighborhoods.shape[0]
    )
)
```

    The dataframe has 5 boroughs and 306 neighborhoods.
    


```python
address = 'New York City, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of New York City are 40.7127281, -74.0060152.
    


```python
# creating map of New York using latitude and longitude values
map_newyork = folium.Map(location=[latitude, longitude], zoom_start=10)

# adding markers to map
for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_newyork)  
    
map_newyork
```

![png](visualizations/NYC.png)
```python
CLIENT_ID = 'JIA5BTQ3NTZCVG2L30RUSA4PIZ1GN5YAVWA2ZJF352ZEOKWL' # your Foursquare ID
CLIENT_SECRET = 'GROWMUMSLS4WSUPAQX1SAMFZFUQPL3YHGG0UYX4IB1Z13SLA' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: JIA5BTQ3NTZCVG2L30RUSA4PIZ1GN5YAVWA2ZJF352ZEOKWL
    CLIENT_SECRET:GROWMUMSLS4WSUPAQX1SAMFZFUQPL3YHGG0UYX4IB1Z13SLA
    

### Exploring Bronx borough for clustering 


```python
bronx_data = neighborhoods[neighborhoods['Borough'] == 'Bronx'].reset_index(drop=True)
bronx_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>Co-op City</td>
      <td>40.874294</td>
      <td>-73.829939</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>Eastchester</td>
      <td>40.887556</td>
      <td>-73.827806</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>Fieldston</td>
      <td>40.895437</td>
      <td>-73.905643</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>Riverdale</td>
      <td>40.890834</td>
      <td>-73.912585</td>
    </tr>
  </tbody>
</table>
</div>




```python
dress = 'Bronx, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Bronx are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Bronx are 40.7127281, -74.0060152.
    


```python
# creating map of Bronx using latitude and longitude values
map_bronx = folium.Map(location=[latitude, longitude], zoom_start=11)

# adding markers to map
for lat, lng, label in zip(bronx_data['Latitude'], bronx_data['Longitude'], bronx_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_bronx)  
    
map_bronx
```

![png](visualizations/bronx.png)







```python
bronx_data.loc[0, 'Neighborhood']
```




    'Wakefield'




```python
neighborhood_latitude = bronx_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = bronx_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = bronx_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))
```

    Latitude and longitude values of Wakefield are 40.89470517661, -73.84720052054902.
    


```python
LIMIT=100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius

# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL

```




    'https://api.foursquare.com/v2/venues/explore?&client_id=JIA5BTQ3NTZCVG2L30RUSA4PIZ1GN5YAVWA2ZJF352ZEOKWL&client_secret=GROWMUMSLS4WSUPAQX1SAMFZFUQPL3YHGG0UYX4IB1Z13SLA&v=20180605&ll=40.89470517661,-73.84720052054902&radius=500&limit=100'




```python
results = requests.get(url).json()
results
```




    {'meta': {'code': 200, 'requestId': '5eb0efa51a4b0a002857b397'},
     'response': {'headerLocation': 'Wakefield',
      'headerFullLocation': 'Wakefield, Bronx',
      'headerLocationGranularity': 'neighborhood',
      'totalResults': 9,
      'suggestedBounds': {'ne': {'lat': 40.899205181110005,
        'lng': -73.84125857127495},
       'sw': {'lat': 40.89020517211, 'lng': -73.8531424698231}},
      'groups': [{'type': 'Recommended Places',
        'name': 'recommended',
        'items': [{'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c537892fd2ea593cb077a28',
           'name': 'Lollipops Gelato',
           'location': {'address': '4120 Baychester Ave',
            'crossStreet': 'Edenwald & Bussing Ave',
            'lat': 40.894123150205274,
            'lng': -73.84589162362325,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.894123150205274,
              'lng': -73.84589162362325},
             {'label': 'entrance', 'lat': 40.89362, 'lng': -73.843737}],
            'distance': 127,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['4120 Baychester Ave (Edenwald & Bussing Ave)',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d1d0941735',
             'name': 'Dessert Shop',
             'pluralName': 'Dessert Shops',
             'shortName': 'Desserts',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/dessert_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c537892fd2ea593cb077a28-0'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c783cef3badb1f7e4244b54',
           'name': 'Carvel Ice Cream',
           'location': {'address': '1006 E 233rd St',
            'lat': 40.890486685759605,
            'lng': -73.84856772568665,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.890486685759605,
              'lng': -73.84856772568665},
             {'label': 'entrance', 'lat': 40.890438, 'lng': -73.848559}],
            'distance': 483,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['1006 E 233rd St',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d1c9941735',
             'name': 'Ice Cream Shop',
             'pluralName': 'Ice Cream Shops',
             'shortName': 'Ice Cream',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/icecream_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c783cef3badb1f7e4244b54-1'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5d5f5044d0ae1c0008f043c3',
           'name': 'Walgreens',
           'location': {'address': '4232 Baychester Ave',
            'crossStreet': 'Pitman',
            'lat': 40.896528,
            'lng': -73.8447,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.896528,
              'lng': -73.8447}],
            'distance': 292,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['4232 Baychester Ave (Pitman)',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d10f951735',
             'name': 'Pharmacy',
             'pluralName': 'Pharmacies',
             'shortName': 'Pharmacy',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/pharmacy_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5d5f5044d0ae1c0008f043c3-2'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d6af9426107f04dedeb297a',
           'name': 'Rite Aid',
           'location': {'address': '4232 Baychester Ave',
            'lat': 40.896649,
            'lng': -73.8448461,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.896649,
              'lng': -73.8448461},
             {'label': 'entrance', 'lat': 40.896504, 'lng': -73.844844}],
            'distance': 293,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['4232 Baychester Ave',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d10f951735',
             'name': 'Pharmacy',
             'pluralName': 'Pharmacies',
             'shortName': 'Pharmacy',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/pharmacy_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d6af9426107f04dedeb297a-3'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c25c212f1272d7f836385c5',
           'name': "Dunkin'",
           'location': {'address': '980 E 233rd St',
            'crossStreet': 'Paulding Ave',
            'lat': 40.8904587811365,
            'lng': -73.84908886747644,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.8904587811365,
              'lng': -73.84908886747644},
             {'label': 'entrance', 'lat': 40.890549, 'lng': -73.849231}],
            'distance': 498,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['980 E 233rd St (Paulding Ave)',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d148941735',
             'name': 'Donut Shop',
             'pluralName': 'Donut Shops',
             'shortName': 'Donuts',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/donuts_',
              'suffix': '.png'},
             'primary': True}],
           'delivery': {'id': '1984066',
            'url': 'https://www.seamless.com/menu/dunkin-980-e-233rd-st-bronx/1984066?affiliate=1131&utm_source=foursquare-affiliate-network&utm_medium=affiliate&utm_campaign=1131&utm_content=1984066',
            'provider': {'name': 'seamless',
             'icon': {'prefix': 'https://fastly.4sqi.net/img/general/cap/',
              'sizes': [40, 50],
              'name': '/delivery_provider_seamless_20180129.png'}}},
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c25c212f1272d7f836385c5-4'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c81a91c51ada1cd87741510',
           'name': 'Shell',
           'location': {'address': '836 E 233rd St',
            'lat': 40.894187118166535,
            'lng': -73.84586195733382,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.894187118166535,
              'lng': -73.84586195733382}],
            'distance': 126,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['836 E 233rd St',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d113951735',
             'name': 'Gas Station',
             'pluralName': 'Gas Stations',
             'shortName': 'Gas Station',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/gas_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c81a91c51ada1cd87741510-5'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d33665fb6093704b80001e0',
           'name': 'Subway',
           'location': {'address': '980 E 233rd St',
            'lat': 40.89046821651127,
            'lng': -73.8491520209833,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.89046821651127,
              'lng': -73.8491520209833}],
            'distance': 499,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['980 E 233rd St',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d1c5941735',
             'name': 'Sandwich Place',
             'pluralName': 'Sandwich Places',
             'shortName': 'Sandwiches',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/deli_',
              'suffix': '.png'},
             'primary': True}],
           'delivery': {'id': '293568',
            'url': 'https://www.seamless.com/menu/subway-980-e-233rd-st-bronx/293568?affiliate=1131&utm_source=foursquare-affiliate-network&utm_medium=affiliate&utm_campaign=1131&utm_content=293568',
            'provider': {'name': 'seamless',
             'icon': {'prefix': 'https://fastly.4sqi.net/img/general/cap/',
              'sizes': [40, 50],
              'name': '/delivery_provider_seamless_20180129.png'}}},
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d33665fb6093704b80001e0-6'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '55aa92ac498e24734cd2e378',
           'name': 'Louis Pizza',
           'location': {'address': '1840 Nereid Ave',
            'crossStreet': 'Ely Ave',
            'lat': 40.898399,
            'lng': -73.84881,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.898399,
              'lng': -73.84881}],
            'distance': 432,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['1840 Nereid Ave (Ely Ave)',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '4bf58dd8d48988d1ca941735',
             'name': 'Pizza Place',
             'pluralName': 'Pizza Places',
             'shortName': 'Pizza',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/pizza_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-55aa92ac498e24734cd2e378-7'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5681717c498e9b9cf4d8c187',
           'name': 'Koss Quick Wash',
           'location': {'address': '951 E 233rd St',
            'crossStreet': 'Edenwald Avenue',
            'lat': 40.891281,
            'lng': -73.84990400000001,
            'labeledLatLngs': [{'label': 'display',
              'lat': 40.891281,
              'lng': -73.84990400000001},
             {'label': 'entrance', 'lat': 40.891288, 'lng': -73.849985}],
            'distance': 443,
            'postalCode': '10466',
            'cc': 'US',
            'city': 'Bronx',
            'state': 'NY',
            'country': 'United States',
            'formattedAddress': ['951 E 233rd St (Edenwald Avenue)',
             'Bronx, NY 10466',
             'United States']},
           'categories': [{'id': '52f2ab2ebcbc57f1066b8b33',
             'name': 'Laundromat',
             'pluralName': 'Laundromats',
             'shortName': 'Laundromat',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/laundry_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5681717c498e9b9cf4d8c187-8'}]}]}}




```python
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
```


```python
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filtering columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filtering the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# cleaning columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lollipops Gelato</td>
      <td>Dessert Shop</td>
      <td>40.894123</td>
      <td>-73.845892</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Carvel Ice Cream</td>
      <td>Ice Cream Shop</td>
      <td>40.890487</td>
      <td>-73.848568</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Walgreens</td>
      <td>Pharmacy</td>
      <td>40.896528</td>
      <td>-73.844700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rite Aid</td>
      <td>Pharmacy</td>
      <td>40.896649</td>
      <td>-73.844846</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dunkin'</td>
      <td>Donut Shop</td>
      <td>40.890459</td>
      <td>-73.849089</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
```

    9 venues were returned by Foursquare.
    


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # creating the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # making the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # returning only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
bronx_venues=getNearbyVenues(names=bronx_data['Neighborhood'],
                                    latitudes=bronx_data['Latitude'],
                                    longitudes=bronx_data['Longitude']
                                  )
```

    Wakefield
    Co-op City
    Eastchester
    Fieldston
    Riverdale
    Kingsbridge
    Woodlawn
    Norwood
    Williamsbridge
    Baychester
    Pelham Parkway
    City Island
    Bedford Park
    University Heights
    Morris Heights
    Fordham
    East Tremont
    West Farms
    High  Bridge
    Melrose
    Mott Haven
    Port Morris
    Longwood
    Hunts Point
    Morrisania
    Soundview
    Clason Point
    Throgs Neck
    Country Club
    Parkchester
    Westchester Square
    Van Nest
    Morris Park
    Belmont
    Spuyten Duyvil
    North Riverdale
    Pelham Bay
    Schuylerville
    Edgewater Park
    Castle Hill
    Olinville
    Pelham Gardens
    Concourse
    Unionport
    Edenwald
    Claremont Village
    Concourse Village
    Mount Eden
    Mount Hope
    Bronxdale
    Allerton
    Kingsbridge Heights
    


```python
print(bronx_venues.shape)
bronx_venues.head()
```

    (1216, 7)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
      <td>Lollipops Gelato</td>
      <td>40.894123</td>
      <td>-73.845892</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
      <td>Carvel Ice Cream</td>
      <td>40.890487</td>
      <td>-73.848568</td>
      <td>Ice Cream Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
      <td>Walgreens</td>
      <td>40.896528</td>
      <td>-73.844700</td>
      <td>Pharmacy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
      <td>Rite Aid</td>
      <td>40.896649</td>
      <td>-73.844846</td>
      <td>Pharmacy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
      <td>Dunkin'</td>
      <td>40.890459</td>
      <td>-73.849089</td>
      <td>Donut Shop</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_venues.groupby('Neighborhood').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Allerton</th>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
    </tr>
    <tr>
      <th>Baychester</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Bedford Park</th>
      <td>37</td>
      <td>37</td>
      <td>37</td>
      <td>37</td>
      <td>37</td>
      <td>37</td>
    </tr>
    <tr>
      <th>Belmont</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Bronxdale</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>Castle Hill</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>City Island</th>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
    </tr>
    <tr>
      <th>Claremont Village</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Clason Point</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Co-op City</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Concourse</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Concourse Village</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Country Club</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>East Tremont</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Eastchester</th>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Edenwald</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Edgewater Park</th>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Fieldston</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Fordham</th>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
    </tr>
    <tr>
      <th>High  Bridge</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Hunts Point</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Kingsbridge</th>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
      <td>66</td>
    </tr>
    <tr>
      <th>Kingsbridge Heights</th>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Longwood</th>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Melrose</th>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
    </tr>
    <tr>
      <th>Morris Heights</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Morris Park</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Morrisania</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Mott Haven</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Mount Eden</th>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Mount Hope</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>North Riverdale</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Norwood</th>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
    </tr>
    <tr>
      <th>Olinville</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>Parkchester</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Pelham Bay</th>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Pelham Gardens</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Pelham Parkway</th>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Port Morris</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Riverdale</th>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>Schuylerville</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Soundview</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Spuyten Duyvil</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>Throgs Neck</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>Unionport</th>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>University Heights</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Van Nest</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>Wakefield</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>West Farms</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Westchester Square</th>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
    </tr>
    <tr>
      <th>Williamsbridge</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Woodlawn</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('There are {} uniques categories.'.format(len(bronx_venues['Venue Category'].unique())))
```

    There are 168 uniques categories.
    


```python
# one hot encoding
bronx_onehot = pd.get_dummies(bronx_venues[['Venue Category']], prefix="", prefix_sep="")

# adding neighborhood column back to dataframe
bronx_onehot['Neighborhood'] = bronx_venues['Neighborhood'] 

# moving neighborhood column to the first column
fixed_columns = [bronx_onehot.columns[-1]] + list(bronx_onehot.columns[:-1])
bronx_onehot = bronx_onehot[fixed_columns]

bronx_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Accessories Store</th>
      <th>African Restaurant</th>
      <th>American Restaurant</th>
      <th>Arcade</th>
      <th>Arepa Restaurant</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>BBQ Joint</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Field</th>
      <th>Basketball Court</th>
      <th>Beer Bar</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Bowling Alley</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Buffet</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Bus Stop</th>
      <th>Café</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Check Cashing Service</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Clothing Store</th>
      <th>Coffee Shop</th>
      <th>Comfort Food Restaurant</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Cupcake Shop</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Distillery</th>
      <th>Dive Bar</th>
      <th>Donut Shop</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Event Space</th>
      <th>Eye Doctor</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Truck</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Furniture / Home Store</th>
      <th>Gas Station</th>
      <th>Gift Shop</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Health &amp; Beauty Service</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Home Service</th>
      <th>Hookah Bar</th>
      <th>Hotel</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Theater</th>
      <th>Intersection</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Juice Bar</th>
      <th>Kids Store</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Laundromat</th>
      <th>Lawyer</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Market</th>
      <th>Martial Arts Dojo</th>
      <th>Mattress Store</th>
      <th>Medical Supply Store</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Mobile Phone Shop</th>
      <th>Music Store</th>
      <th>Music Venue</th>
      <th>Nightclub</th>
      <th>Outdoor Sculpture</th>
      <th>Outlet Store</th>
      <th>Paella Restaurant</th>
      <th>Paper / Office Supplies Store</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Peruvian Restaurant</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Piano Bar</th>
      <th>Pizza Place</th>
      <th>Platform</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Pool</th>
      <th>Pub</th>
      <th>Recreation Center</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>Seafood Restaurant</th>
      <th>Shipping Store</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Smoke Shop</th>
      <th>Snack Place</th>
      <th>Social Club</th>
      <th>Soup Place</th>
      <th>South American Restaurant</th>
      <th>South Indian Restaurant</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Spanish Restaurant</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Sports Club</th>
      <th>Storage Facility</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Tattoo Parlor</th>
      <th>Tennis Court</th>
      <th>Tennis Stadium</th>
      <th>Thai Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Track</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Waste Facility</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wakefield</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wakefield</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wakefield</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wakefield</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wakefield</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_onehot.shape
```




    (1216, 169)




```python
bronx_grouped = bronx_onehot.groupby('Neighborhood').mean().reset_index()
bronx_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Accessories Store</th>
      <th>African Restaurant</th>
      <th>American Restaurant</th>
      <th>Arcade</th>
      <th>Arepa Restaurant</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>BBQ Joint</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Field</th>
      <th>Basketball Court</th>
      <th>Beer Bar</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Bowling Alley</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Buffet</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Bus Line</th>
      <th>Bus Station</th>
      <th>Bus Stop</th>
      <th>Café</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Check Cashing Service</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Clothing Store</th>
      <th>Coffee Shop</th>
      <th>Comfort Food Restaurant</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Cupcake Shop</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Distillery</th>
      <th>Dive Bar</th>
      <th>Donut Shop</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Event Space</th>
      <th>Eye Doctor</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Truck</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Furniture / Home Store</th>
      <th>Gas Station</th>
      <th>Gift Shop</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Health &amp; Beauty Service</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hobby Shop</th>
      <th>Home Service</th>
      <th>Hookah Bar</th>
      <th>Hotel</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Theater</th>
      <th>Intersection</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Juice Bar</th>
      <th>Kids Store</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Laundromat</th>
      <th>Lawyer</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Market</th>
      <th>Martial Arts Dojo</th>
      <th>Mattress Store</th>
      <th>Medical Supply Store</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Mobile Phone Shop</th>
      <th>Music Store</th>
      <th>Music Venue</th>
      <th>Nightclub</th>
      <th>Outdoor Sculpture</th>
      <th>Outlet Store</th>
      <th>Paella Restaurant</th>
      <th>Paper / Office Supplies Store</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Peruvian Restaurant</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Piano Bar</th>
      <th>Pizza Place</th>
      <th>Platform</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Pool</th>
      <th>Pub</th>
      <th>Recreation Center</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>Seafood Restaurant</th>
      <th>Shipping Store</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Smoke Shop</th>
      <th>Snack Place</th>
      <th>Social Club</th>
      <th>Soup Place</th>
      <th>South American Restaurant</th>
      <th>South Indian Restaurant</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Spanish Restaurant</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Sports Club</th>
      <th>Storage Facility</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Tattoo Parlor</th>
      <th>Tennis Court</th>
      <th>Tennis Stadium</th>
      <th>Thai Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Track</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Waste Facility</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allerton</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.00</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.03125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.062500</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.00</td>
      <td>0.156250</td>
      <td>0.00</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Baychester</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.105263</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.052632</td>
      <td>0.00</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.052632</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bedford Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.108108</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.081081</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.108108</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.081081</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.00</td>
      <td>0.108108</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.054054</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Belmont</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.010000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.190000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.01</td>
      <td>0.090000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.020000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.010000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronxdale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.076923</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Castle Hill</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.00</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>City Island</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.074074</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.037037</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.074074</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.074074</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Claremont Village</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.105263</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.105263</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.157895</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.105263</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Clason Point</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Co-op City</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.00</td>
      <td>0.066667</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Concourse</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.160000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00</td>
      <td>0.080000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Concourse Village</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.00</td>
      <td>0.028571</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.085714</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.028571</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Country Club</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>East Tremont</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.157895</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Eastchester</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.150000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Edenwald</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.142857</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Edgewater Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.150000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.100000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Fieldston</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Fordham</td>
      <td>0.012821</td>
      <td>0.012821</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.051282</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012821</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.051282</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.012821</td>
      <td>0.025641</td>
      <td>0.038462</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.012821</td>
      <td>0.012821</td>
      <td>0.00</td>
      <td>0.012821</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.064103</td>
      <td>0.012821</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>0.00</td>
      <td>0.051282</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.025641</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.051282</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012821</td>
      <td>0.038462</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>High  Bridge</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.00</td>
      <td>0.125000</td>
      <td>0.00</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Hunts Point</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Kingsbridge</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.030303</td>
      <td>0.015152</td>
      <td>0.060606</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.030303</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.030303</td>
      <td>0.00</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.015152</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015152</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.015152</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Kingsbridge Heights</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.00</td>
      <td>0.166667</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000</td>
      <td>0.033333</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Longwood</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.125000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Melrose</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.074074</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.074074</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.00</td>
      <td>0.148148</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.074074</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037037</td>
      <td>0.037037</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Morris Heights</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.00</td>
      <td>0.111111</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Morris Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.041667</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00</td>
      <td>0.208333</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Morrisania</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00</td>
      <td>0.080000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.040000</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Mott Haven</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.00</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Mount Eden</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.068966</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.103448</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.103448</td>
      <td>0.00</td>
      <td>0.103448</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.034483</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.068966</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.068966</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Mount Hope</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.090909</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>North Riverdale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.125000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Norwood</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.093750</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.093750</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.062500</td>
      <td>0.00</td>
      <td>0.156250</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000</td>
      <td>0.031250</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Olinville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.153846</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.153846</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Parkchester</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.057143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.085714</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000</td>
      <td>0.028571</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.114286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.057143</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Pelham Bay</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.025000</td>
      <td>0.025000</td>
      <td>0.075000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.075000</td>
      <td>0.00</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.025</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025</td>
      <td>0.050000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Pelham Gardens</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.095238</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.095238</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.095238</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.047619</td>
      <td>0.047619</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.095238</td>
      <td>0.00</td>
      <td>0.047619</td>
      <td>0.00</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.047619</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Pelham Parkway</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.107143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.035714</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Port Morris</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Riverdale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.181818</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Schuylerville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.090909</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Soundview</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Spuyten Duyvil</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.100</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Throgs Neck</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.153846</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.153846</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.076923</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Unionport</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.086957</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.086957</td>
      <td>0.043478</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.086957</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.043478</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>University Heights</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.041667</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00</td>
      <td>0.125000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.041667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Van Nest</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.105263</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.105263</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.210526</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Wakefield</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.222222</td>
      <td>0.00</td>
      <td>0.111111</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.111111</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>West Farms</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.227273</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Westchester Square</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.00</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.093750</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.03125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.03125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.00</td>
      <td>0.062500</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.062500</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Williamsbridge</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Woodlawn</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00</td>
      <td>0.160000</td>
      <td>0.00</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_grouped.shape
```




    (52, 169)




```python
num_top_venues = 5

for hood in bronx_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = bronx_grouped[bronx_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ----Allerton----
                    venue  freq
    0         Pizza Place  0.16
    1  Chinese Restaurant  0.06
    2         Supermarket  0.06
    3              Bakery  0.06
    4    Department Store  0.06
    
    
    ----Baychester----
                      venue  freq
    0            Donut Shop  0.11
    1  Fast Food Restaurant  0.05
    2   Fried Chicken Joint  0.05
    3           Supermarket  0.05
    4             Pet Store  0.05
    
    
    ----Bedford Park----
                    venue  freq
    0               Diner  0.11
    1         Pizza Place  0.11
    2  Chinese Restaurant  0.11
    3       Deli / Bodega  0.08
    4  Mexican Restaurant  0.08
    
    
    ----Belmont----
                    venue  freq
    0  Italian Restaurant  0.19
    1         Pizza Place  0.09
    2       Deli / Bodega  0.08
    3              Bakery  0.05
    4          Donut Shop  0.03
    
    
    ----Bronxdale----
                             venue  freq
    0           Spanish Restaurant  0.08
    1                         Park  0.08
    2           Italian Restaurant  0.08
    3  Eastern European Restaurant  0.08
    4                  Supermarket  0.08
    
    
    ----Castle Hill----
                venue  freq
    0     Pizza Place  0.25
    1            Bank  0.12
    2  Baseball Field  0.12
    3   Deli / Bodega  0.12
    4           Diner  0.12
    
    
    ----City Island----
                        venue  freq
    0         Harbor / Marina  0.11
    1      Seafood Restaurant  0.07
    2  Thrift / Vintage Store  0.07
    3           Boat or Ferry  0.07
    4          Ice Cream Shop  0.04
    
    
    ----Claremont Village----
                    venue  freq
    0       Grocery Store  0.16
    1         Pizza Place  0.11
    2  Chinese Restaurant  0.11
    3         Bus Station  0.11
    4           Gift Shop  0.05
    
    
    ----Clason Point----
               venue  freq
    0           Park   0.4
    1           Pool   0.1
    2       Bus Stop   0.1
    3            Spa   0.1
    4  Boat or Ferry   0.1
    
    
    ----Co-op City----
                venue  freq
    0  Baseball Field  0.13
    1     Bus Station  0.13
    2  Discount Store  0.07
    3      Bagel Shop  0.07
    4            Park  0.07
    
    
    ----Concourse----
               venue  freq
    0  Grocery Store  0.16
    1    Pizza Place  0.08
    2         Bakery  0.08
    3     Donut Shop  0.04
    4  Deli / Bodega  0.04
    
    
    ----Concourse Village----
                      venue  freq
    0        Sandwich Place  0.09
    1    Mexican Restaurant  0.06
    2         Deli / Bodega  0.06
    3  Fast Food Restaurant  0.06
    4              Pharmacy  0.06
    
    
    ----Country Club----
                     venue  freq
    0       Sandwich Place  0.50
    1           Playground  0.25
    2   Athletics & Sports  0.25
    3    Accessories Store  0.00
    4  Peruvian Restaurant  0.00
    
    
    ----East Tremont----
                   venue  freq
    0        Pizza Place  0.16
    1        Flea Market  0.05
    2         Shoe Store  0.05
    3           Bus Stop  0.05
    4  Paella Restaurant  0.05
    
    
    ----Eastchester----
                      venue  freq
    0  Caribbean Restaurant  0.15
    1                 Diner  0.10
    2         Deli / Bodega  0.10
    3         Metro Station  0.05
    4    Seafood Restaurant  0.05
    
    
    ----Edenwald----
                    venue  freq
    0         Gas Station  0.14
    1         Supermarket  0.14
    2         Pizza Place  0.14
    3       Grocery Store  0.14
    4  Athletics & Sports  0.14
    
    
    ----Edgewater Park----
                    venue  freq
    0  Italian Restaurant  0.20
    1       Deli / Bodega  0.15
    2         Pizza Place  0.10
    3                 Bar  0.05
    4        Liquor Store  0.05
    
    
    ----Fieldston----
                     venue  freq
    0                Plaza   0.5
    1          Bus Station   0.5
    2    Accessories Store   0.0
    3  Peruvian Restaurant   0.0
    4            Nightclub   0.0
    
    
    ----Fordham----
                   venue  freq
    0  Mobile Phone Shop  0.06
    1               Bank  0.05
    2         Donut Shop  0.05
    3         Shoe Store  0.05
    4        Pizza Place  0.05
    
    
    ----High  Bridge----
                venue  freq
    0     Pizza Place  0.12
    1        Pharmacy  0.12
    2   Deli / Bodega  0.08
    3  Sandwich Place  0.04
    4   Metro Station  0.04
    
    
    ----Hunts Point----
                    venue  freq
    0  Spanish Restaurant  0.08
    1         Pizza Place  0.08
    2       Grocery Store  0.08
    3                Bank  0.08
    4      Farmers Market  0.08
    
    
    ----Kingsbridge----
                           venue  freq
    0                Pizza Place  0.09
    1                        Bar  0.06
    2                Supermarket  0.05
    3         Mexican Restaurant  0.05
    4  Latin American Restaurant  0.05
    
    
    ----Kingsbridge Heights----
                    venue  freq
    0         Pizza Place  0.17
    1  Chinese Restaurant  0.10
    2       Grocery Store  0.07
    3         Coffee Shop  0.07
    4  Mexican Restaurant  0.07
    
    
    ----Longwood----
                           venue  freq
    0              Deli / Bodega  0.12
    1                  Wine Shop  0.12
    2  Latin American Restaurant  0.12
    3                      Diner  0.12
    4             Sandwich Place  0.12
    
    
    ----Melrose----
                venue  freq
    0     Pizza Place  0.15
    1        Pharmacy  0.11
    2  Discount Store  0.07
    3   Grocery Store  0.07
    4  Sandwich Place  0.07
    
    
    ----Morris Heights----
                           venue  freq
    0          Recreation Center  0.11
    1         Spanish Restaurant  0.11
    2  Latin American Restaurant  0.11
    3                       Park  0.11
    4                Pizza Place  0.11
    
    
    ----Morris Park----
               venue  freq
    0    Pizza Place  0.21
    1  Deli / Bodega  0.08
    2         Bakery  0.08
    3   Burger Joint  0.08
    4    Supermarket  0.04
    
    
    ----Morrisania----
                      venue  freq
    0        Discount Store  0.12
    1         Metro Station  0.08
    2           Pizza Place  0.08
    3  Fast Food Restaurant  0.08
    4           Bus Station  0.08
    
    
    ----Mott Haven----
                           venue  freq
    0                Pizza Place  0.09
    1         Spanish Restaurant  0.09
    2                 Donut Shop  0.09
    3                        Gym  0.09
    4  Latin American Restaurant  0.05
    
    
    ----Mount Eden----
                      venue  freq
    0              Pharmacy  0.10
    1           Pizza Place  0.10
    2  Fast Food Restaurant  0.10
    3    Spanish Restaurant  0.07
    4           Supermarket  0.07
    
    
    ----Mount Hope----
                     venue  freq
    0        Grocery Store  0.18
    1  Fried Chicken Joint  0.09
    2        Metro Station  0.09
    3        Deli / Bodega  0.09
    4           Donut Shop  0.09
    
    
    ----North Riverdale----
                    venue  freq
    0         Pizza Place  0.12
    1  Italian Restaurant  0.08
    2                Bank  0.08
    3         Bus Station  0.04
    4    Sushi Restaurant  0.04
    
    
    ----Norwood----
             venue  freq
    0  Pizza Place  0.16
    1         Park  0.12
    2  Bus Station  0.09
    3         Bank  0.09
    4     Pharmacy  0.06
    
    
    ----Olinville----
                        venue  freq
    0             Supermarket  0.15
    1    Caribbean Restaurant  0.15
    2           Metro Station  0.08
    3        Basketball Court  0.08
    4  Furniture / Home Store  0.08
    
    
    ----Parkchester----
                     venue  freq
    0          Supermarket  0.11
    1          Pizza Place  0.09
    2        Women's Store  0.06
    3  American Restaurant  0.06
    4           Kids Store  0.06
    
    
    ----Pelham Bay----
                      venue  freq
    0    Italian Restaurant  0.08
    1                  Bank  0.08
    2  Fast Food Restaurant  0.05
    3     Convenience Store  0.05
    4  Gym / Fitness Center  0.05
    
    
    ----Pelham Gardens----
                    venue  freq
    0         Bus Station  0.10
    1          Donut Shop  0.10
    2  Chinese Restaurant  0.10
    3            Pharmacy  0.10
    4        Intersection  0.05
    
    
    ----Pelham Parkway----
                    venue  freq
    0         Bus Station  0.11
    1  Frozen Yogurt Shop  0.07
    2  Italian Restaurant  0.07
    3  Chinese Restaurant  0.07
    4         Pizza Place  0.07
    
    
    ----Port Morris----
                           venue  freq
    0  Latin American Restaurant  0.13
    1     Furniture / Home Store  0.13
    2             Clothing Store  0.07
    3                 Restaurant  0.07
    4                 Donut Shop  0.07
    
    
    ----Riverdale----
              venue  freq
    0   Bus Station  0.18
    1          Park  0.18
    2  Home Service  0.09
    3    Food Truck  0.09
    4           Gym  0.09
    
    
    ----Schuylerville----
                    venue  freq
    0               Diner  0.09
    1                Bank  0.09
    2         Pizza Place  0.09
    3            Pharmacy  0.09
    4  Mexican Restaurant  0.09
    
    
    ----Soundview----
                    venue  freq
    0  Chinese Restaurant  0.20
    1       Grocery Store  0.13
    2      Discount Store  0.07
    3            Bus Stop  0.07
    4        Burger Joint  0.07
    
    
    ----Spuyten Duyvil----
              venue  freq
    0          Park   0.2
    1  Intersection   0.1
    2      Pharmacy   0.1
    3          Food   0.1
    4  Tennis Court   0.1
    
    
    ----Throgs Neck----
               venue  freq
    0  Deli / Bodega  0.15
    1    Coffee Shop  0.15
    2      Juice Bar  0.08
    3            Bar  0.08
    4    Pizza Place  0.08
    
    
    ----Unionport----
                           venue  freq
    0  Latin American Restaurant  0.09
    1             Ice Cream Shop  0.09
    2                 Donut Shop  0.09
    3          Electronics Store  0.04
    4         Chinese Restaurant  0.04
    
    
    ----University Heights----
                     venue  freq
    0          Pizza Place  0.12
    1   Chinese Restaurant  0.12
    2  Fried Chicken Joint  0.08
    3       Sandwich Place  0.04
    4       History Museum  0.04
    
    
    ----Van Nest----
               venue  freq
    0    Pizza Place  0.21
    1         Bakery  0.11
    2  Deli / Bodega  0.11
    3     Donut Shop  0.05
    4            Spa  0.05
    
    
    ----Wakefield----
                venue  freq
    0        Pharmacy  0.22
    1    Dessert Shop  0.11
    2     Gas Station  0.11
    3  Sandwich Place  0.11
    4  Ice Cream Shop  0.11
    
    
    ----West Farms----
                  venue  freq
    0       Bus Station  0.23
    1              Park  0.09
    2        Donut Shop  0.09
    3      Intersection  0.05
    4  Basketball Court  0.05
    
    
    ----Westchester Square----
                      venue  freq
    0  Fast Food Restaurant  0.09
    1           Pizza Place  0.06
    2        Sandwich Place  0.06
    3            Donut Shop  0.06
    4              Pharmacy  0.06
    
    
    ----Williamsbridge----
                      venue  freq
    0  Caribbean Restaurant  0.33
    1            Soup Place  0.17
    2           Supermarket  0.17
    3             Nightclub  0.17
    4                   Bar  0.17
    
    
    ----Woodlawn----
               venue  freq
    0    Pizza Place  0.16
    1            Bar  0.08
    2     Playground  0.08
    3  Deli / Bodega  0.08
    4           Park  0.04
    
    
    


```python
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```


```python
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# creating columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# creating a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = bronx_grouped['Neighborhood']

for ind in np.arange(bronx_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(bronx_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Allerton</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Department Store</td>
      <td>Deli / Bodega</td>
      <td>Supermarket</td>
      <td>Chinese Restaurant</td>
      <td>Spa</td>
      <td>Fried Chicken Joint</td>
      <td>Breakfast Spot</td>
      <td>Spanish Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Baychester</td>
      <td>Donut Shop</td>
      <td>Spanish Restaurant</td>
      <td>Bus Station</td>
      <td>Men's Store</td>
      <td>Pizza Place</td>
      <td>Fast Food Restaurant</td>
      <td>Mattress Store</td>
      <td>Supermarket</td>
      <td>Fried Chicken Joint</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bedford Park</td>
      <td>Pizza Place</td>
      <td>Diner</td>
      <td>Chinese Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Mexican Restaurant</td>
      <td>Sandwich Place</td>
      <td>Donut Shop</td>
      <td>Smoke Shop</td>
      <td>Burger Joint</td>
      <td>Spanish Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Belmont</td>
      <td>Italian Restaurant</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Dessert Shop</td>
      <td>Grocery Store</td>
      <td>Donut Shop</td>
      <td>Mexican Restaurant</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronxdale</td>
      <td>Pizza Place</td>
      <td>Gym</td>
      <td>Spanish Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Supermarket</td>
      <td>Eastern European Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Paper / Office Supplies Store</td>
      <td>Bank</td>
      <td>Park</td>
    </tr>
  </tbody>
</table>
</div>




```python
# setting number of clusters
kclusters = 5

bronx_grouped_clustering = bronx_grouped.drop('Neighborhood', 1)

# running k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(bronx_grouped_clustering)

# checking cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([3, 3, 3, 3, 3, 3, 3, 0, 0, 0])




```python
# adding clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

bronx_merged = bronx_data

# merging toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
bronx_merged = bronx_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

bronx_merged.head() # checking the last columns!
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bronx</td>
      <td>Wakefield</td>
      <td>40.894705</td>
      <td>-73.847201</td>
      <td>0</td>
      <td>Pharmacy</td>
      <td>Ice Cream Shop</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Laundromat</td>
      <td>Donut Shop</td>
      <td>Gas Station</td>
      <td>Event Space</td>
      <td>Fish Market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bronx</td>
      <td>Co-op City</td>
      <td>40.874294</td>
      <td>-73.829939</td>
      <td>0</td>
      <td>Bus Station</td>
      <td>Baseball Field</td>
      <td>Ice Cream Shop</td>
      <td>Park</td>
      <td>Fast Food Restaurant</td>
      <td>Mattress Store</td>
      <td>Chinese Restaurant</td>
      <td>Bagel Shop</td>
      <td>Grocery Store</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bronx</td>
      <td>Eastchester</td>
      <td>40.887556</td>
      <td>-73.827806</td>
      <td>1</td>
      <td>Caribbean Restaurant</td>
      <td>Diner</td>
      <td>Deli / Bodega</td>
      <td>Platform</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Metro Station</td>
      <td>Bowling Alley</td>
      <td>Fast Food Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bronx</td>
      <td>Fieldston</td>
      <td>40.895437</td>
      <td>-73.905643</td>
      <td>2</td>
      <td>Plaza</td>
      <td>Bus Station</td>
      <td>Women's Store</td>
      <td>Event Space</td>
      <td>Food</td>
      <td>Flea Market</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bronx</td>
      <td>Riverdale</td>
      <td>40.890834</td>
      <td>-73.912585</td>
      <td>3</td>
      <td>Park</td>
      <td>Bus Station</td>
      <td>Gym</td>
      <td>Food Truck</td>
      <td>Medical Supply Store</td>
      <td>Baseball Field</td>
      <td>Bank</td>
      <td>Plaza</td>
      <td>Home Service</td>
      <td>Donut Shop</td>
    </tr>
  </tbody>
</table>
</div>




```python
# creating map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# setting color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# adding markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(bronx_merged['Latitude'], bronx_merged['Longitude'], bronx_merged['Neighborhood'], bronx_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




![png](visualizations/clustering_bronx.png)



```python
bronx_merged.loc[bronx_merged['Cluster Labels'] == 0, bronx_merged.columns[[1] + list(range(5, bronx_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wakefield</td>
      <td>Pharmacy</td>
      <td>Ice Cream Shop</td>
      <td>Dessert Shop</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>Laundromat</td>
      <td>Donut Shop</td>
      <td>Gas Station</td>
      <td>Event Space</td>
      <td>Fish Market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Co-op City</td>
      <td>Bus Station</td>
      <td>Baseball Field</td>
      <td>Ice Cream Shop</td>
      <td>Park</td>
      <td>Fast Food Restaurant</td>
      <td>Mattress Store</td>
      <td>Chinese Restaurant</td>
      <td>Bagel Shop</td>
      <td>Grocery Store</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>13</th>
      <td>University Heights</td>
      <td>Pizza Place</td>
      <td>Chinese Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Cosmetics Shop</td>
      <td>Bakery</td>
      <td>Burger Joint</td>
      <td>Shoe Store</td>
      <td>Supermarket</td>
      <td>Sandwich Place</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Morris Heights</td>
      <td>Chinese Restaurant</td>
      <td>Pizza Place</td>
      <td>Recreation Center</td>
      <td>Pharmacy</td>
      <td>Park</td>
      <td>Latin American Restaurant</td>
      <td>Grocery Store</td>
      <td>Spanish Restaurant</td>
      <td>Bank</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Melrose</td>
      <td>Pizza Place</td>
      <td>Pharmacy</td>
      <td>Discount Store</td>
      <td>Grocery Store</td>
      <td>Sandwich Place</td>
      <td>Intersection</td>
      <td>Clothing Store</td>
      <td>Mexican Restaurant</td>
      <td>Department Store</td>
      <td>Martial Arts Dojo</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Mott Haven</td>
      <td>Spanish Restaurant</td>
      <td>Pizza Place</td>
      <td>Gym</td>
      <td>Donut Shop</td>
      <td>Department Store</td>
      <td>Bakery</td>
      <td>Burger Joint</td>
      <td>Mobile Phone Shop</td>
      <td>Storage Facility</td>
      <td>Bookstore</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Longwood</td>
      <td>Deli / Bodega</td>
      <td>Sandwich Place</td>
      <td>Fast Food Restaurant</td>
      <td>Grocery Store</td>
      <td>Latin American Restaurant</td>
      <td>Donut Shop</td>
      <td>Diner</td>
      <td>Wine Shop</td>
      <td>Event Space</td>
      <td>Flea Market</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Morrisania</td>
      <td>Discount Store</td>
      <td>Metro Station</td>
      <td>Fast Food Restaurant</td>
      <td>Liquor Store</td>
      <td>Donut Shop</td>
      <td>Bus Station</td>
      <td>Pizza Place</td>
      <td>Pharmacy</td>
      <td>Mexican Restaurant</td>
      <td>Fish Market</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Soundview</td>
      <td>Chinese Restaurant</td>
      <td>Grocery Store</td>
      <td>Liquor Store</td>
      <td>Discount Store</td>
      <td>Bus Station</td>
      <td>Pharmacy</td>
      <td>Bus Stop</td>
      <td>Video Store</td>
      <td>Breakfast Spot</td>
      <td>Fried Chicken Joint</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Clason Point</td>
      <td>Park</td>
      <td>Pool</td>
      <td>South American Restaurant</td>
      <td>Boat or Ferry</td>
      <td>Bus Stop</td>
      <td>Grocery Store</td>
      <td>Spa</td>
      <td>Farmers Market</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Pelham Gardens</td>
      <td>Donut Shop</td>
      <td>Chinese Restaurant</td>
      <td>Bus Station</td>
      <td>Pharmacy</td>
      <td>Spanish Restaurant</td>
      <td>Bank</td>
      <td>Playground</td>
      <td>Deli / Bodega</td>
      <td>Sandwich Place</td>
      <td>Bus Line</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Concourse</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Rental Car Location</td>
      <td>Bus Station</td>
      <td>Metro Station</td>
      <td>Pharmacy</td>
      <td>Spanish Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Fried Chicken Joint</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Edenwald</td>
      <td>Food</td>
      <td>Grocery Store</td>
      <td>Fish Market</td>
      <td>Pizza Place</td>
      <td>Supermarket</td>
      <td>Gas Station</td>
      <td>Athletics &amp; Sports</td>
      <td>Flea Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Claremont Village</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Chinese Restaurant</td>
      <td>Bus Station</td>
      <td>Discount Store</td>
      <td>Bakery</td>
      <td>Food</td>
      <td>Fried Chicken Joint</td>
      <td>Caribbean Restaurant</td>
      <td>Gift Shop</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Mount Eden</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Pizza Place</td>
      <td>Spanish Restaurant</td>
      <td>Supermarket</td>
      <td>Deli / Bodega</td>
      <td>Mobile Phone Shop</td>
      <td>Chinese Restaurant</td>
      <td>Food Truck</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Mount Hope</td>
      <td>Grocery Store</td>
      <td>Ice Cream Shop</td>
      <td>Sandwich Place</td>
      <td>Donut Shop</td>
      <td>Spanish Restaurant</td>
      <td>Metro Station</td>
      <td>Fried Chicken Joint</td>
      <td>Supermarket</td>
      <td>Deli / Bodega</td>
      <td>Video Game Store</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_merged.loc[bronx_merged['Cluster Labels'] == 1, bronx_merged.columns[[1] + list(range(5, bronx_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Eastchester</td>
      <td>Caribbean Restaurant</td>
      <td>Diner</td>
      <td>Deli / Bodega</td>
      <td>Platform</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Metro Station</td>
      <td>Bowling Alley</td>
      <td>Fast Food Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Williamsbridge</td>
      <td>Caribbean Restaurant</td>
      <td>Supermarket</td>
      <td>Nightclub</td>
      <td>Soup Place</td>
      <td>Bar</td>
      <td>Eye Doctor</td>
      <td>Flea Market</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Olinville</td>
      <td>Supermarket</td>
      <td>Caribbean Restaurant</td>
      <td>Metro Station</td>
      <td>Laundromat</td>
      <td>Fried Chicken Joint</td>
      <td>Furniture / Home Store</td>
      <td>Basketball Court</td>
      <td>Chinese Restaurant</td>
      <td>Liquor Store</td>
      <td>Convenience Store</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_merged.loc[bronx_merged['Cluster Labels'] == 2, bronx_merged.columns[[1] + list(range(5, bronx_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Fieldston</td>
      <td>Plaza</td>
      <td>Bus Station</td>
      <td>Women's Store</td>
      <td>Event Space</td>
      <td>Food</td>
      <td>Flea Market</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_merged.loc[bronx_merged['Cluster Labels'] == 3, bronx_merged.columns[[1] + list(range(5, bronx_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Riverdale</td>
      <td>Park</td>
      <td>Bus Station</td>
      <td>Gym</td>
      <td>Food Truck</td>
      <td>Medical Supply Store</td>
      <td>Baseball Field</td>
      <td>Bank</td>
      <td>Plaza</td>
      <td>Home Service</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kingsbridge</td>
      <td>Pizza Place</td>
      <td>Bar</td>
      <td>Mexican Restaurant</td>
      <td>Latin American Restaurant</td>
      <td>Sandwich Place</td>
      <td>Supermarket</td>
      <td>Pharmacy</td>
      <td>Spanish Restaurant</td>
      <td>Bakery</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Woodlawn</td>
      <td>Pizza Place</td>
      <td>Playground</td>
      <td>Deli / Bodega</td>
      <td>Bar</td>
      <td>Grocery Store</td>
      <td>Pharmacy</td>
      <td>Pub</td>
      <td>Convenience Store</td>
      <td>Rental Car Location</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Norwood</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Bus Station</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Deli / Bodega</td>
      <td>Chinese Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Pet Store</td>
      <td>Spanish Restaurant</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Baychester</td>
      <td>Donut Shop</td>
      <td>Spanish Restaurant</td>
      <td>Bus Station</td>
      <td>Men's Store</td>
      <td>Pizza Place</td>
      <td>Fast Food Restaurant</td>
      <td>Mattress Store</td>
      <td>Supermarket</td>
      <td>Fried Chicken Joint</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Pelham Parkway</td>
      <td>Bus Station</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Frozen Yogurt Shop</td>
      <td>Bank</td>
      <td>Mexican Restaurant</td>
      <td>Metro Station</td>
      <td>Eye Doctor</td>
      <td>Food</td>
    </tr>
    <tr>
      <th>11</th>
      <td>City Island</td>
      <td>Harbor / Marina</td>
      <td>Seafood Restaurant</td>
      <td>Boat or Ferry</td>
      <td>Thrift / Vintage Store</td>
      <td>Baseball Field</td>
      <td>Grocery Store</td>
      <td>French Restaurant</td>
      <td>Music Venue</td>
      <td>Diner</td>
      <td>Spanish Restaurant</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Bedford Park</td>
      <td>Pizza Place</td>
      <td>Diner</td>
      <td>Chinese Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Mexican Restaurant</td>
      <td>Sandwich Place</td>
      <td>Donut Shop</td>
      <td>Smoke Shop</td>
      <td>Burger Joint</td>
      <td>Spanish Restaurant</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Fordham</td>
      <td>Mobile Phone Shop</td>
      <td>Pizza Place</td>
      <td>Bank</td>
      <td>Donut Shop</td>
      <td>Shoe Store</td>
      <td>Gym / Fitness Center</td>
      <td>Spanish Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Supplement Shop</td>
      <td>Pharmacy</td>
    </tr>
    <tr>
      <th>16</th>
      <td>East Tremont</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Shoe Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Spanish Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Mobile Phone Shop</td>
      <td>Bus Stop</td>
      <td>Supermarket</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>17</th>
      <td>West Farms</td>
      <td>Bus Station</td>
      <td>Donut Shop</td>
      <td>Park</td>
      <td>Bus Stop</td>
      <td>Diner</td>
      <td>Coffee Shop</td>
      <td>Basketball Court</td>
      <td>Lounge</td>
      <td>Metro Station</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>18</th>
      <td>High  Bridge</td>
      <td>Pharmacy</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Latin American Restaurant</td>
      <td>Metro Station</td>
      <td>Spanish Restaurant</td>
      <td>Check Cashing Service</td>
      <td>Sports Club</td>
      <td>Market</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Port Morris</td>
      <td>Furniture / Home Store</td>
      <td>Latin American Restaurant</td>
      <td>Brewery</td>
      <td>Clothing Store</td>
      <td>Food Truck</td>
      <td>Storage Facility</td>
      <td>Music Venue</td>
      <td>Donut Shop</td>
      <td>Spanish Restaurant</td>
      <td>Peruvian Restaurant</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Hunts Point</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Spanish Restaurant</td>
      <td>Café</td>
      <td>Farmers Market</td>
      <td>Shipping Store</td>
      <td>Gourmet Shop</td>
      <td>Bank</td>
      <td>BBQ Joint</td>
      <td>Grocery Store</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Throgs Neck</td>
      <td>Coffee Shop</td>
      <td>Deli / Bodega</td>
      <td>Asian Restaurant</td>
      <td>Mobile Phone Shop</td>
      <td>Sports Bar</td>
      <td>Baseball Field</td>
      <td>Bar</td>
      <td>Pizza Place</td>
      <td>Juice Bar</td>
      <td>American Restaurant</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Parkchester</td>
      <td>Supermarket</td>
      <td>Pizza Place</td>
      <td>Women's Store</td>
      <td>American Restaurant</td>
      <td>Kids Store</td>
      <td>Department Store</td>
      <td>Shoe Store</td>
      <td>Caribbean Restaurant</td>
      <td>Sandwich Place</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Westchester Square</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>Pharmacy</td>
      <td>Pizza Place</td>
      <td>Donut Shop</td>
      <td>Mobile Phone Shop</td>
      <td>Building</td>
      <td>Metro Station</td>
      <td>Mexican Restaurant</td>
      <td>Check Cashing Service</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Van Nest</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Bakery</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Donut Shop</td>
      <td>Bus Station</td>
      <td>Middle Eastern Restaurant</td>
      <td>Spa</td>
      <td>Food Truck</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Morris Park</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Burger Joint</td>
      <td>Bakery</td>
      <td>Bank</td>
      <td>Donut Shop</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Food</td>
      <td>Buffet</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Belmont</td>
      <td>Italian Restaurant</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Bakery</td>
      <td>Café</td>
      <td>Dessert Shop</td>
      <td>Grocery Store</td>
      <td>Donut Shop</td>
      <td>Mexican Restaurant</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Spuyten Duyvil</td>
      <td>Park</td>
      <td>Thai Restaurant</td>
      <td>Bank</td>
      <td>Tennis Court</td>
      <td>Tennis Stadium</td>
      <td>Scenic Lookout</td>
      <td>Food</td>
      <td>Pharmacy</td>
      <td>Intersection</td>
      <td>Discount Store</td>
    </tr>
    <tr>
      <th>35</th>
      <td>North Riverdale</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Pool</td>
      <td>Sushi Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Donut Shop</td>
      <td>Electronics Store</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Pelham Bay</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Convenience Store</td>
      <td>Fast Food Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Gym / Fitness Center</td>
      <td>Sandwich Place</td>
      <td>Donut Shop</td>
      <td>Diner</td>
      <td>Lawyer</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Schuylerville</td>
      <td>Pizza Place</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>American Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Diner</td>
      <td>Convenience Store</td>
      <td>Donut Shop</td>
      <td>Sandwich Place</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Edgewater Park</td>
      <td>Italian Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Pizza Place</td>
      <td>Bar</td>
      <td>Donut Shop</td>
      <td>Liquor Store</td>
      <td>Coffee Shop</td>
      <td>Sports Bar</td>
      <td>Park</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Castle Hill</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Bank</td>
      <td>Diner</td>
      <td>Pharmacy</td>
      <td>Baseball Field</td>
      <td>Market</td>
      <td>Flea Market</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Unionport</td>
      <td>Ice Cream Shop</td>
      <td>Donut Shop</td>
      <td>Latin American Restaurant</td>
      <td>Dance Studio</td>
      <td>Pizza Place</td>
      <td>Deli / Bodega</td>
      <td>Plaza</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Comfort Food Restaurant</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Concourse Village</td>
      <td>Sandwich Place</td>
      <td>Sporting Goods Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Pharmacy</td>
      <td>Deli / Bodega</td>
      <td>Park</td>
      <td>Chinese Restaurant</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Bronxdale</td>
      <td>Pizza Place</td>
      <td>Gym</td>
      <td>Spanish Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Supermarket</td>
      <td>Eastern European Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Paper / Office Supplies Store</td>
      <td>Bank</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Allerton</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Department Store</td>
      <td>Deli / Bodega</td>
      <td>Supermarket</td>
      <td>Chinese Restaurant</td>
      <td>Spa</td>
      <td>Fried Chicken Joint</td>
      <td>Breakfast Spot</td>
      <td>Spanish Restaurant</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Kingsbridge Heights</td>
      <td>Pizza Place</td>
      <td>Chinese Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Grocery Store</td>
      <td>Coffee Shop</td>
      <td>Pharmacy</td>
      <td>Latin American Restaurant</td>
      <td>Metro Station</td>
      <td>Café</td>
      <td>Spanish Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_merged.loc[bronx_merged['Cluster Labels'] == 4, bronx_merged.columns[[1] + list(range(5, bronx_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>Country Club</td>
      <td>Sandwich Place</td>
      <td>Playground</td>
      <td>Athletics &amp; Sports</td>
      <td>Women's Store</td>
      <td>Event Space</td>
      <td>Food</td>
      <td>Flea Market</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
bronx_merged.loc[bronx_merged['Cluster Labels'] == 5, bronx_merged.columns[[1] + list(range(5, bronx_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
restaurants_neighborhood=bronx_merged.loc[bronx_merged['Cluster Labels'] == 3, bronx_merged.columns[[1] + list(range(5, bronx_merged.shape[1]))]]
```


```python
type(restaurants_neighborhood)
```




    pandas.core.frame.DataFrame




```python
df=restaurants_neighborhood[restaurants_neighborhood['1st Most Common Venue'] == 'Chinese Restaurant']
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



**Most of the restaurants could be found in cluster 0 and cluster 3. However, in  cluster 0, there weren't any Chinese  restaurants which were the first most common venue. In cluster 3 however, in Soundview neighborhood, Chinese Restaurant is the most common venue and this tells us that Chinese food is prominent in this area. Thus, opening up a restaurant in this neighborhood would be a profitable choice** 


```python

```
