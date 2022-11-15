# f2da26b9-b8f1-450d-89a7-d4643ce560d1
https://sonarcloud.io/summary/overall?id=examly-test_f2da26b9-b8f1-450d-89a7-d4643ce560d1



# Stock Market Crash Prediction in 2022

## Project Summary
This project deals with the Prediction of upcoming crashes in 2022 along with the confidence score and the reason for crash.

## Lessons Learned

This project deals with many parameters which are indirectly contributing to crash. The difficulty I faced was during data collection and correlation analysis among them. I learnt to analyze the and apply accurate techniques to get the results precisely.

Crash : A crash refers to a sudden rapid loss in value of the market, which can last for months or years. The temporary and short time drops are not considered as crash, with standing and considerably long time of drop of value leads to crash of stock market.
## Tech Stack

**Client:** Python

**Server:** Streamlit


## Table of Contents

- Data Information & Collection
- Additional Parameters
- Model Selection
- Prediction of Crash
- Web Interface


### Data Information & Collection

The crucial entity of this project is data collection. The data of 6 Countries are used to predict the crashes for Indian Stock Market. The countries USA, Japan, HongKong, China and Brazil are taken for analysis. They are from over past 30 years of sensex data. 

### Additional Parameters

The additional parameters like GDP, Unemployment rate etc. are used to analyze the reasons for crash. Few other crash contributing factors can be considered to describe the crash. There are more than 1000 factors which directly or indirectly cause the crash.

### Model Selection

The Model selection is the major part after the data collection and preprossing.
It deals with the modeling of data and gaining the insights from them. 
There were many models which can be used to predict the current project scenario. 
Linear Regression is used to give the confidence score or the probability of occurance of crash. 
Between Linear Regression and Logistic Regression , Linear Regression had the better accuracy and more informatiove than Logistic Regression.

### Prediciton Of Crash

The crash prediction is based on the threshold values which are calculated from the data of all 6 countries.
The drawdown values are calculated for every year and then they lead to give information of length of crash in previous data.
The beta and threshold values will give the tolerance of every year to be predicted.
These threshold values are given to model to predict the crash in terms of number of months from current year.

### Image
Linear vs Logistic
https://drive.google.com/file/d/1RbTawSkSLKtyfEOr881OGGztiUpu-jCS/view?usp=share_link

### Web Interface

The Web Interface is developed using streamlit platform. 
This gives both internal and external network to access the interface. 
The results from the model are captured/collected and then displayed on the interface.
The interface consists of previous crash and then the expected crash along with crash probabilities being highlighted.

### Image
Interface
https://drive.google.com/file/d/1ftjcvZHy4C1t1OGlao_e1fX6HXs-I06u/view?usp=share_link

### Conclusion

The model predict the crash after 18 months with 0.56 confidence score and after 24 months with 0.97 accuracy. This concludes that the crash will not occur in 2022. One of reasons for crash in future may be due to the recession in current period which shows a long term effect. There may be other contributing hidden factors which cause the crash in future.