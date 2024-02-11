#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import plotly.express as px

warnings.filterwarnings('ignore')


# In[2]:


df_fdi = pd.read_csv(r'D:\Internship Project\FDI data.csv')


# Getting one with data

# In[3]:


df_fdi.head(5)


# In[4]:


df_fdi.info()


# In[5]:


df_fdi.Sector.unique()


# In[6]:


df_fdi.Sector.nunique()


# In[7]:


df_fdi.isnull().sum()


# In[8]:


df_fdi.duplicated().sum()


# Visualizing the dataset

# In[9]:


df_fdi_new = df_fdi.set_index('Sector')


# In[10]:


df_fdi_trans = df_fdi_new.T
df_fdi_trans.head()


# # ENERGY SECTOR

# In[11]:


df_fdi_energy = df_fdi_trans[['METALLURGICAL INDUSTRIES', 'MINING', 'POWER',
       'NON-CONVENTIONAL ENERGY', 'COAL PRODUCTION', 'PETROLEUM & NATURAL GAS',
       'BOILERS AND STEAM GENERATING PLANTS']]
df_fdi_energy.head(4)


# In[12]:


df_fdi_energy.sum()


# In[13]:


df_fdi_energy.max()


# In[14]:


df_fdi_energy.boxplot();
plt.xticks(rotation='90')
plt.show()


# In[15]:


plt.pie(df_fdi_energy.sum(),labels=['METALLURGICAL INDUSTRIES', 'MINING', 'POWER',
       'NON-CONVENTIONAL ENERGY', 'COAL PRODUCTION', 'PETROLEUM & NATURAL GAS',
       'BOILERS AND STEAM GENERATING PLANTS'], shadow=True, explode=[0.2,0.2,0.2,0.2,0.2,0.2,0.2], autopct="%1.1f%%");


# In[16]:


plt.figure(figsize=(8,8))
sns.lineplot(x = df_fdi_energy.index, y = df_fdi_energy['METALLURGICAL INDUSTRIES'], data= df_fdi_energy, label='METALLURGICAL INDUSTRIES')
sns.lineplot(x = df_fdi_energy.index, y = df_fdi_energy['MINING'], data= df_fdi_energy, label = 'Mining')
sns.lineplot(x = df_fdi_energy.index, y = df_fdi_energy['POWER'], data= df_fdi_energy,label = 'Power')
sns.lineplot(x = df_fdi_energy.index, y = df_fdi_energy['PETROLEUM & NATURAL GAS'], data= df_fdi_energy, label= 'Petro')
sns.lineplot(x = df_fdi_energy.index, y = df_fdi_energy['NON-CONVENTIONAL ENERGY'], data= df_fdi_energy, label = 'Non Conventional Energy')

plt.xlabel('Years')
plt.ylabel('FDI in Energy Industries')
plt.xticks(rotation = '90')
plt.show()


# In[17]:


import plotly.express as px

fig = px.line(
    df_fdi_energy,
    x=df_fdi_energy.index,
    y=["METALLURGICAL INDUSTRIES", "MINING", "POWER", "PETROLEUM & NATURAL GAS", "NON-CONVENTIONAL ENERGY"],
    labels={"x": "Years", "y": "FDI in Mining Industries"},
    title="FDI in Energy Sectors over Time",
)

fig.update_layout(
    xaxis_title="Years",
    yaxis_title="FDI in Energy Industries",
    xaxis_tickangle=-45,  
    hovermode="x unified",  
)

fig.show()


# # TRANSPORT SECTOR

# In[18]:


df_fdi_transport = df_fdi_trans[['AUTOMOBILE INDUSTRY', 'AIR TRANSPORT (INCLUDING AIR FREIGHT)',
       'SEA TRANSPORT', 'PORTS', 'RAILWAY RELATED COMPONENTS']]
df_fdi_transport.head(4)


# In[19]:


df_fdi_transport.sum()


# In[20]:


plt.pie(df_fdi_transport.sum(), labels = df_fdi_transport.columns, explode = [0.2,0.2,0.2,0.2,0.2], shadow = True, autopct="%1.1f%%");


# In[21]:


plt.figure(figsize=(6,4))
sns.lineplot(x=df_fdi_transport.index, y= df_fdi_transport['SEA TRANSPORT'], data = df_fdi_transport, label='Sea')
sns.lineplot(x=df_fdi_transport.index, y= df_fdi_transport['PORTS'], data = df_fdi_transport, label='Ports')
sns.lineplot(x=df_fdi_transport.index, y= df_fdi_transport['AIR TRANSPORT (INCLUDING AIR FREIGHT)'], data = df_fdi_transport, label='Airways')
sns.lineplot(x=df_fdi_transport.index, y= df_fdi_transport['RAILWAY RELATED COMPONENTS'], data = df_fdi_transport, label='Railways')
sns.lineplot(x=df_fdi_transport.index, y= df_fdi_transport['AUTOMOBILE INDUSTRY'], data = df_fdi_transport, label='Roadways')
plt.xlabel('Years')
plt.ylabel('FDI in Transport')
plt.xticks(rotation = '90')
plt.show()


# # EQUIPMENTS AND INSTRUMENTS

# In[22]:


df_fdi_equip = df_fdi_trans[['COMMERCIAL, OFFICE & HOUSEHOLD EQUIPMENTS',
       'MEDICAL AND SURGICAL APPLIANCES', 'INDUSTRIAL INSTRUMENTS',
       'SCIENTIFIC INSTRUMENTS',
       'MATHEMATICAL,SURVEYING AND DRAWING INSTRUMENTS']]
df_fdi_equip.head(3)


# In[23]:


df_fdi_equip.sum()


# In[24]:


plt.pie(df_fdi_equip.sum(), labels = df_fdi_equip.columns, explode = [0.2,0.2,0.2,0.2,0.2], shadow =True, autopct='%1.1f%%');


# In[25]:


plt.figure(figsize=(6,4))
sns.lineplot(x = df_fdi_transport.index, y = df_fdi_transport['AUTOMOBILE INDUSTRY'], data= df_fdi_transport, label='Automobile')
sns.lineplot(x = df_fdi_transport.index, y = df_fdi_transport['AIR TRANSPORT (INCLUDING AIR FREIGHT)'], data= df_fdi_transport, label = 'Airlines')
sns.lineplot(x = df_fdi_transport.index, y = df_fdi_transport['SEA TRANSPORT'], data= df_fdi_transport,label = 'Sea Transport')
sns.lineplot(x = df_fdi_transport.index, y = df_fdi_transport['PORTS'], data= df_fdi_transport, label= 'Ports')
sns.lineplot(x = df_fdi_transport.index, y = df_fdi_transport['RAILWAY RELATED COMPONENTS'], data= df_fdi_transport, label = 'Railways')

plt.xlabel('Years')
plt.ylabel('FDI in Transport Sector')
plt.xticks(rotation = '90')
plt.show()


# # ELECTRONICS AND IT SECTOR

# In[26]:


df_fdi_it = df_fdi_trans[['ELECTRICAL EQUIPMENTS', 'COMPUTER SOFTWARE & HARDWARE', 'ELECTRONICS','TELECOMMUNICATIONS',
       'INFORMATION & BROADCASTING (INCLUDING PRINT MEDIA)']]
df_fdi_it.head(3)


# In[27]:


#total FDI in each sectors
df_fdi_it.sum()


# In[28]:


plt.pie(df_fdi_it.sum(), labels = df_fdi_it.columns, explode = [0.2,0.2,0.2,0.2,0.2], shadow=True, autopct="%1.1f%%");


# In[29]:


plt.figure(figsize=(6,6))
sns.lineplot(x = df_fdi_it.index, y = df_fdi_it['COMPUTER SOFTWARE & HARDWARE'], data=df_fdi_it, label='COMPUTER SOFTWARE & HARDWARE')
sns.lineplot(x = df_fdi_it.index, y = df_fdi_it['INFORMATION & BROADCASTING (INCLUDING PRINT MEDIA)'], data=df_fdi_it, label = 'INFORMATION & BROADCASTING')
sns.lineplot(x = df_fdi_it.index, y = df_fdi_it['TELECOMMUNICATIONS'], data=df_fdi_it, label = 'TELECOMMUNICATIONS')
sns.lineplot(x = df_fdi_it.index, y = df_fdi_it['ELECTRONICS'], data=df_fdi_it, label = 'ELECTRONICS')
sns.lineplot(x = df_fdi_it.index, y = df_fdi_it['ELECTRICAL EQUIPMENTS'], data=df_fdi_it, label = 'ELECTRICAL EQUIPMENTS')
plt.xlabel('Years')
plt.ylabel('FDI in IT Sector')
plt.xticks(rotation = '90')
plt.legend()
plt.show()


# # HEAVY INDUSTRIAL GOODS

# In[30]:


df_fdi_hig = df_fdi_trans[['INDUSTRIAL MACHINERY', 'MACHINE TOOLS', 'AGRICULTURAL MACHINERY',
       'EARTH-MOVING MACHINERY',
       'MISCELLANEOUS MECHANICAL & ENGINEERING INDUSTRIES']]
df_fdi_hig.head(4)


# In[31]:


df_fdi_hig.sum()


# In[32]:


plt.pie(df_fdi_hig.sum(), labels = df_fdi_hig.columns, explode = [0.2,0.2,0.2,0.2,0.2], shadow = True, autopct='%1.1f%%');


# In[33]:


plt.figure(figsize=(6,6))
sns.lineplot(x=df_fdi_hig.index, y= df_fdi_hig['MACHINE TOOLS'], data=df_fdi_hig, label='Machine Tools')
sns.lineplot(x=df_fdi_hig.index, y= df_fdi_hig['AGRICULTURAL MACHINERY'], data=df_fdi_hig, label='Agricultural tools')
sns.lineplot(x=df_fdi_hig.index, y= df_fdi_hig['EARTH-MOVING MACHINERY'], data=df_fdi_hig, label='Earth Moving Machines')
sns.lineplot(x=df_fdi_hig.index, y= df_fdi_hig['INDUSTRIAL MACHINERY'], data=df_fdi_hig, label='Industrial Machine')
sns.lineplot(x=df_fdi_hig.index, y= df_fdi_hig['MISCELLANEOUS MECHANICAL & ENGINEERING INDUSTRIES'], data=df_fdi_hig, label='Miscellaneous')
plt.xlabel('Years')
plt.ylabel('FDI in Heavy Industrial Goods')
plt.xticks(rotation='90')
plt.show()


# # AGRICULTURE SECTOR

# In[34]:


df_fdi_agri = df_fdi_trans[['SUGAR',
       'FERMENTATION INDUSTRIES', 'FOOD PROCESSING INDUSTRIES',
       'VEGETABLE OILS AND VANASPATI','FERTILIZERS']]
df_fdi_agri.head(3)


# In[35]:


df_fdi_agri.sum()


# In[36]:


plt.pie(df_fdi_agri.sum(), labels = df_fdi_agri.columns, explode = [0.2,0.2,0.2,0.2,0.2], shadow = True, autopct='%1.1f%%');


# In[37]:


plt.figure(figsize=(6,6))
sns.lineplot(x=df_fdi_agri.index, y= df_fdi_agri['FOOD PROCESSING INDUSTRIES'], data=df_fdi_agri, label='FPI')
sns.lineplot(x=df_fdi_agri.index, y= df_fdi_agri['FERTILIZERS'], data=df_fdi_agri, label='Fertilizer')
sns.lineplot(x=df_fdi_agri.index, y= df_fdi_agri['FERMENTATION INDUSTRIES'], data=df_fdi_agri, label='Fermentation')
sns.lineplot(x=df_fdi_agri.index, y= df_fdi_agri['VEGETABLE OILS AND VANASPATI'], data=df_fdi_agri, label='Edible oil')
sns.lineplot(x=df_fdi_agri.index, y= df_fdi_agri['SUGAR'], data=df_fdi_agri, label='Sugar')
plt.xlabel('Years')
plt.ylabel('FDI in Agri')
plt.xticks(rotation='90')
plt.show()


# # CONSUMER GOODS

# In[38]:


df_fdi_consumer = df_fdi_trans[['SOAPS, COSMETICS & TOILET PREPARATIONS', 'RUBBER GOODS',
       'LEATHER,LEATHER GOODS AND PICKERS', 'GLUE AND GELATIN', 'GLASS',
       'CERAMICS', 'CEMENT AND GYPSUM PRODUCTS', 'TIMBER PRODUCTS','TEXTILES (INCLUDING DYED,PRINTED)']]
df_fdi_consumer.head(3)


# In[39]:


df_fdi_consumer.sum()


# In[40]:


plt.pie(df_fdi_consumer.sum(), labels = df_fdi_consumer.columns, explode = [0.1,0.4,0.4,0.2,0.1,0.2,0.2,0.1,0.1], shadow=True, autopct='%1.1f%%');


# In[41]:


plt.figure(figsize=(6,6))
sns.lineplot(x=df_fdi_consumer.index, y= df_fdi_consumer['CEMENT AND GYPSUM PRODUCTS'], data=df_fdi_consumer, label='Cement')
sns.lineplot(x=df_fdi_consumer.index, y= df_fdi_consumer['TEXTILES (INCLUDING DYED,PRINTED)'], data=df_fdi_consumer, label='Textile')
sns.lineplot(x=df_fdi_consumer.index, y= df_fdi_consumer['SOAPS, COSMETICS & TOILET PREPARATIONS'], data=df_fdi_consumer, label='Cosmetics & Soaps')
sns.lineplot(x=df_fdi_consumer.index, y= df_fdi_consumer['RUBBER GOODS'], data=df_fdi_consumer, label='Rubber')
sns.lineplot(x=df_fdi_consumer.index, y= df_fdi_consumer['TIMBER PRODUCTS'], data=df_fdi_consumer, label='Timber')
plt.xlabel('Years')
plt.ylabel('FDI in Consumer Goods')
plt.xticks(rotation='90')
plt.show()


# # SERVICES SECTOR

# In[42]:


df_fdi_services = df_fdi_trans[['DEFENCE INDUSTRIES', 'CONSULTANCY SERVICES',
       'SERVICES SECTOR (Fin.,Banking,Insurance,Non Fin/Business,Outsourcing,R&D,Courier,Tech. Testing and Analysis, Other)',
       'HOSPITAL & DIAGNOSTIC CENTRES', 'EDUCATION', 'HOTEL & TOURISM',
       'TRADING', 'RETAIL TRADING', 'AGRICULTURE SERVICES']]

new_names = {'SERVICES SECTOR (Fin.,Banking,Insurance,Non Fin/Business,Outsourcing,R&D,Courier,Tech. Testing and Analysis, Other)': 'FINANCIAL SERVICES'}
df_fdi_services.rename(columns=new_names, inplace=True)  

df_fdi_services.head(3)


# In[43]:


df_fdi_services.sum()


# In[44]:


plt.pie(df_fdi_services.sum(), labels = df_fdi_services.columns, explode = [0.1,0.2,0.3,0.1,0.2,0.1,0.2,0.4,0.4], shadow = True, autopct='%1.1f%%');


# # CAGR

# In[45]:


#Calculating Compound Annual Growth Rate
df_fdi_copy = df_fdi.copy()
df_fdi_copy['CAGR'] = (((df_fdi_copy['2016-17']/df_fdi_copy['2000-01'])**1/17)-1)*100
df_fdi_copy[['Sector','CAGR','2000-01']]


# In[46]:


df_fdi_copy.head()


# In[47]:


print(df_fdi_copy['CAGR'].to_string())


# IT CAN BE SEEN THAT FEW OF THE ROWS HAVE VALUE INFINITE. IT IS BECAUSE OF 0 IN THE PERIOD 2000-01

# In[48]:


df_fdi_copy = df_fdi_copy[~np.isinf(df_fdi_copy['CAGR'])]
df_fdi_copy['CAGR']


# In[49]:


plt.figure(figsize=(15,6))
sns.lineplot(x=df_fdi_copy.Sector,y=df_fdi_copy.CAGR, data=df_fdi_copy)
plt.xticks(rotation='90')
plt.show()


# In[50]:


df_fdi_copy[df_fdi_copy['Sector']=='RUBBER GOODS']


# # Arima model on the metallurgical industries

# In[51]:


plt.plot(df_fdi_trans.index, df_fdi_trans['METALLURGICAL INDUSTRIES'] )
plt.xticks(rotation = '90')
plt.show()


# In[52]:


df_fdi_trans['METALLURGICAL INDUSTRIES'].max(), df_fdi_trans['METALLURGICAL INDUSTRIES'].min()


# In[53]:


fig = px.line(
    df_fdi_trans,
    x=df_fdi_trans.index,
    y="METALLURGICAL INDUSTRIES",
    title="Metallurgical Industries FDI over Time",
    width = 1000,
    height = 800
)
fig.show()


# In[54]:


#checking for stationarity of data
from statsmodels.tsa.stattools import adfuller

def adfuller_test(sector):
    adfuller_result = adfuller(sector, autolag=None)
    adfuller_out = pd.Series(adfuller_result[0:4],
                            index=['Test Statistics','p-value','Lags Used','Number of observation'])
    print(adfuller_out)


# In[55]:


adfuller_test(df_fdi_trans['METALLURGICAL INDUSTRIES'])


# Since p>0.05, the data is non stationary, thus using differencing to convert it into stationary

# In[56]:


df_fdi_trans['METALLURGICAL INDUSTRIES Diff'] = df_fdi_trans['METALLURGICAL INDUSTRIES'] - df_fdi_trans['METALLURGICAL INDUSTRIES'].shift(1)
df_fdi_trans[['METALLURGICAL INDUSTRIES','METALLURGICAL INDUSTRIES Diff']].head()


# In[57]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df_fdi_trans['METALLURGICAL INDUSTRIES Diff'].dropna(), lags=10);


# Since the ACF plot reaches to zero fairly quick, then the value of d = 1

# In[58]:


#for q value, we will use ACF plot as well
plot_acf(df_fdi_trans['METALLURGICAL INDUSTRIES'], lags = 10);


# Since there is one value significantly above the significant area, then MA(1) i.e q = 1

# In[59]:


#for p we will plot pacf graph
plot_pacf(df_fdi_trans['METALLURGICAL INDUSTRIES'], lags= 7);


# Since thre are two values for which pacf is above significant area, then p =2

# In[60]:


df_metallurgy_train = df_fdi_trans['METALLURGICAL INDUSTRIES'][0:]


# In[61]:


from statsmodels.tsa.arima.model import ARIMA

arima = ARIMA(df_metallurgy_train.astype(np.float64).values,
             order=(2,1,1))
arima_model = arima.fit()


# In[62]:


arima_model.summary()


# In[63]:


plot_acf(arima_model.resid, lags=10);


# Since there is not auto correlation of residuals, so the model can be assumed to be valid

# In[64]:


metallurgy_predict = arima_model.forecast(steps = 1)
metallurgy_predict


# In[65]:


def get_mape(actual,predict):
    y_true,y_pred=np.array(actual),np.array(predict)
    return np.round(np.mean(np.abs((actual-predict)/actual))*100,2)


# In[66]:


get_mape(df_fdi_trans['METALLURGICAL INDUSTRIES'][16:],metallurgy_predict )


# In[67]:


metallurgy_predict = arima_model.forecast(steps = 4)
metallurgy_predict
df_pred_metallurgy = pd.DataFrame({'index':['2016-17','2017-18','2018-19','2019-20'],
                                  'METALLURGICAL INDUSTRIES':[1492.24300526,  916.52761103,  713.02159702,  998.14093847]})
df_pred_metallurgy


# In[68]:


import seaborn as sns

sns.lineplot(x=df_fdi_trans.index, y=df_fdi_trans['METALLURGICAL INDUSTRIES'], data=df_fdi_trans['METALLURGICAL INDUSTRIES'], label="Series 1")
sns.lineplot(x="index", y="METALLURGICAL INDUSTRIES", data=df_pred_metallurgy, label="Series 2")
plt.title("Comparison of Series 1 and Series 2")
plt.xlabel('Years')
plt.ylabel('FDI in Metallurgical Industries')
plt.xticks(rotation = '90')
plt.show()


# In[69]:


df_fdi_trans['METALLURGICAL INDUSTRIES'].sum()


# In[70]:


def CAGR(present,past,num_of_years):
    Com_ann_growth_rate = (((present/past)**1/num_of_years)-1)*100
    return Com_ann_growth_rate


# In[71]:


cagr_metallurgy = round(CAGR(1440.18,22.69,17),3)
print(f'The Compound Annual Growth Rate for mettalurgy is {cagr_metallurgy}%')

cagr_metallurgy_predicted = round(CAGR(998,1492,4),3)
print(f'The Predicted CAGR for FY20 with respect to FY17 is {cagr_metallurgy_predicted}%')


# # ARIMA ON MINNING INDUSTRY

# In[72]:


df_fdi_trans['MINING']


# In[73]:


sns.lineplot(x=df_fdi_trans['MINING'].index, y=df_fdi_trans['MINING'])
plt.xticks(rotation='90')
plt.show()


# In[74]:


#checking for outliers
def clean_df(dataset):
    mean = dataset.mean()
    std = dataset.std()
    z_scores = (dataset - mean) / std

    outliers = dataset[(abs(z_scores) > 3) | (z_scores < -3) ]
    
    cleaned_df = dataset[~dataset.isin(outliers)]
    
    return pd.DataFrame({'Z-Score':z_scores,'FDI':dataset})


# In[75]:


clean_df(df_fdi_trans['MINING'])


# In[76]:


#checking for stationarity
adfuller_test(df_fdi_trans['MINING'])


# thus data is not stationary, as p>0.05

# In[77]:


#Differencing
df_fdi_trans['MINING DIFF'] = df_fdi_trans['MINING'] - df_fdi_trans['MINING'].shift(1)
df_fdi_trans[['MINING','MINING DIFF']].head(5)


# In[78]:


#checking for stationarity
plot_acf(df_fdi_trans['MINING DIFF'].dropna(), lags =10);


# d = 1, as there is sebsequent decrease in acf value

# In[79]:


#calculating q
plot_acf(df_fdi_trans['MINING'], lags=10);


# q=1 , i.e moving average is 1.

# In[80]:


#calculating p
plot_pacf(df_fdi_trans['MINING'], lags=7);


# p=1

# In[81]:


#building model
arima_mining = ARIMA(df_fdi_trans['MINING'].astype(np.float64).values,
                    order = (2,1,1))
arima_mining_model = arima_mining.fit()
arima_mining_model.summary()


# In[82]:


#checking for correlation of residual
plot_acf(arima_mining_model.resid, lags =10);


# lack of correlation between the residual that can be seen in the graph

# In[83]:


mining_predict = arima_mining_model.forecast(steps=1)
mining_predict


# In[84]:


get_mape(df_fdi_trans['MINING'], mining_predict)


# In[85]:


arima_mining_model.forecast(steps=4)


# In[86]:


df_pred_mining = pd.DataFrame({'index':['2016-17','2017-18','2018-19','2019-20'],
                                  'MINING':[194.85234748, 328.11190284, 284.43351737, 246.32036714]})
df_pred_mining


# In[87]:


sns.lineplot(x=df_fdi_trans.index, y=df_fdi_trans['MINING'], data=df_fdi_trans['MINING'], label="Series 1")
sns.lineplot(x="index", y="MINING", data=df_pred_mining, label="Series 2")
plt.title("Comparison of Series 1 and Series 2")
plt.xlabel('Years')
plt.ylabel('FDI in Mining Industries')
plt.xticks(rotation = '90')
plt.show()


# In[88]:


cagr_mining = CAGR(df_fdi_trans['MINING'][16], df_fdi_trans['MINING'][0], 17)
print(f'The Compound Annual Growth Rate for mettalurgy is {cagr_mining}%')


# # ARIMA On RUBBER GOODS

# In[89]:


df_fdi_trans['RUBBER GOODS']


# In[90]:


sns.lineplot(x=df_fdi_trans['RUBBER GOODS'].index, y=df_fdi_trans['RUBBER GOODS'])
plt.xticks(rotation='90')
plt.show()


# In[91]:


clean_df(df_fdi_trans['RUBBER GOODS'])


# In[92]:


#check for stationarity
adfuller_test(df_fdi_trans['RUBBER GOODS'])


# p>0.05, thus it is non stationary

# In[93]:


#Differencing
df_fdi_trans['RUBBER GOODS DIFF'] = df_fdi_trans['RUBBER GOODS'] - df_fdi_trans['RUBBER GOODS'].shift(1)
df_fdi_trans[['RUBBER GOODS','RUBBER GOODS DIFF']].head(5)


# In[94]:


#checking for stationarity
plot_acf(df_fdi_trans['RUBBER GOODS DIFF'].dropna(), lags =3);


# d=1

# In[95]:


#calculating q
plot_acf(df_fdi_trans['RUBBER GOODS'], lags=5);


# q=2

# In[96]:


#calculating p
plot_pacf(df_fdi_trans['RUBBER GOODS'], lags=5);


# p=2

# In[97]:


#building model
arima_rubber = ARIMA(df_fdi_trans['RUBBER GOODS'].astype(np.float64).values,
                    order = (2,1,2))
arima_rubber_model = arima_rubber.fit()
arima_rubber_model.summary()


# In[98]:


#checking for correlation of residual
plot_acf(arima_rubber_model.resid, lags =10);


# In[99]:


rubber_predict = arima_rubber_model.forecast(steps=4)
rubber_predict


# In[100]:


df_pred_rubber = pd.DataFrame({'index':['2016-17','2017-18','2018-19','2019-20'],
                                  'Rubber':[282.48140842, 261.45259   , 279.64504985, 262.14552046]})
df_pred_rubber


# In[101]:


sns.lineplot(x=df_fdi_trans.index, y=df_fdi_trans['RUBBER GOODS'], data=df_fdi_trans['RUBBER GOODS'], label="Series 1")
sns.lineplot(x="index", y="Rubber", data=df_pred_rubber, label="Series 2")
plt.title("Comparison of Series 1 and Series 2")
plt.xlabel('Years')
plt.ylabel('FDI in Rubber Industries')
plt.xticks(rotation = '90')
plt.show()


# In[102]:


get_mape(262.76,282.48)


# # comparison between sectors

# In[103]:


#use df_fdi_trans['sector'] to get result
def get_comparison(sector1,sector2,sector3=None):
    sns.lineplot(x=sector1.index,y=sector1, data=df_fdi_trans, label= 'Sector1')
    sns.lineplot(x=sector2.index,y=sector2, data=df_fdi_trans, label = 'Sector2')
    if sector3 is not None:
        sns.lineplot(x=sector3.index, y=sector3, data=df_fdi_trans, label='Sector 3')
    plt.xlabel('Years')
    plt.ylabel('FDI')
    plt.xticks(rotation='90')
    plt.show


# In[104]:


get_comparison(df_fdi_trans['MINING'], df_fdi_trans['POWER'], df_fdi_trans['AUTOMOBILE INDUSTRY'])

