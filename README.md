# Sector_wise_FDI
This dataset comprises foreign direct investment (FDI) into 63 sectors across 17 years, ranging from 2000 to 2017.

![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/4d74daa0-5c66-470f-8ab0-096940ef052b)

## Getting One with the Data
### Sectors
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/a05b97f3-ff96-44ab-a54c-22f79444e27a)

### Number of Sectors
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/042248e8-838b-40e7-bc31-4a68e7e3cf49)

### Data Cleaning - removing null and duplicated data
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/f31df10a-7560-48f7-afc2-1653f7c09133)

## Further classifying Sector wise data
63 sectors are further broadly grouped into :-
1. Energy Sector
2. Transport Sector
3. Equipments and Instruments
4. Electronics & IT sectr
5. Heavy Industrial Goods
6. Agriculture Sector
7. Consumer Goods
8. Services Sector


# Understanding Each Broad Classification

In order t0 understand sub sectors, the data has been transposed.
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/f2d1771e-b306-46e9-86ce-8c859c6b63bc)

All these broader classification has been cleaned,visualized and transformed in a similar way.
## 1. Energy Sector
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/130aa39b-c40c-4369-a5d8-75e037432b37)

### Total FDI In Energy Sector
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/9efd0c1f-688b-4c83-bc4b-322ea7ae5742)

### Sub Sector Maximum FDI
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/7b6f5647-609e-4bd0-ad97-7554d0c612a1)

### Sub- Sector Division of the Energy Sector's FDI
It can be visualized using pie chart. This chart showcases the percentage of the FDI sub-sector wise w.r.t Total FDI in the sector. 
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/02b87cb2-ad61-411c-a5e0-4d30acc5e237)

### Sub Sector Yearly Growth
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/ae42b041-06fc-4a0b-9b47-e4d38a2ae4e1)

Interactive Plot - Using plotly.express
![newplot (2)](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/715a4387-7de4-4ee7-8bf7-a30c9ffca05a)

## Feature Extraction - Extracting new features depending on the available columns
### 1. CAGR - Compounded Annual Growth Rate
Compound Annual Growth Rate (CAGR) is a metric used to measure the average annual growth rate of an investment over a specific period, taking into account the compounding effect. This means that it considers how reinvesting your earnings can potentially accelerate your overall growth.

CAGR = (Ending Value / Beginning Value)^(1/Number of Periods) - 1

![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/324ea985-0068-42de-a46d-32ec349b7a87)

There are some sector with values like infinite and NaN, mainly because of the presence of '0'.
These values shall be removed in order to visualize and understand the data.

![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/116b7730-aa15-44b6-bfff-7b19060fd115)

# Using ML to forecast the FDI in the sector
## ARIMA Model =
The ARIMA model stands for Autoregressive Integrated Moving Average and is a powerful statistical tool used for forecasting future values in time series data. It captures the essential patterns and trends within the data, allowing you to make informed predictions about what might happen next.

## Steps to be followed in ARIMA model:-
1. Checking for the stationarity in the data <br>
-> Using adfuller test. If p > 0.05, then it is non-stationary <br>
   ![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/8f34f5e0-f459-4c08-94be-c6d798395762)

2. Use of differencing to induce stationary. It can be done
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/92a4ebd6-01cb-401a-a63e-53a03d7507ca)
<br>

In ARIMA models, differencing plays a crucial role in preparing non-stationary time series data for accurate forecasting.

Stationarity:

ARIMA models require stationary data, where the statistical properties (mean, variance, etc.) remain constant over time.
Non-stationary data often exhibits trends or seasonality, making forecasting unreliable.
Differencing removes these trends and seasonality by calculating the difference between consecutive data points.
This essentially transforms the data into a more stable form, where the statistical properties hold, allowing ARIMA models to effectively analyze and predict future values.

3. Calculatin p,d,q for ARIMA model
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/1f1002e4-4800-45b6-981b-f566faff3a33)

![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/7ccb3911-895e-4e00-a79c-44b97243333a)

4. Fitting ARIMA model
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/ca5ea8e5-ccad-4e39-8648-fe210513f2ab)

5. Predict
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/0979f79e-a54b-4935-8dba-099be6856219)

6. Calculating Error
![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/2067ae7e-81f0-4a7e-b26a-941d6e918f0c)
<br>
MAPE:-
Mean Absolute Percentage Error, is a popular metric used to assess the accuracy of forecasts. It measures the average percentage difference between predicted and actual values, providing an easily interpretable way to understand how close your forecasts are to reality.

<br>
7. Visualizing Prediction
<br>

![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/9355cac2-0c87-4a0b-b0bb-b05011a38be6)

<br>

# Comparison between sectors
<br>
A function "get_comparison" helps to get compared visualized data between two sectors
<br>

![image](https://github.com/shashank-2010/Sector_wise_FDI/assets/153171192/eda11342-ed2e-42e8-b5c4-6fcb38b98a2c)

<br>


























