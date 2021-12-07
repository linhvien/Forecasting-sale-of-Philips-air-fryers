# Forecasting-sale-of-Philips-air-fryers
I apply time-series analysis and machine learning model to forecast the sale of Philips air fryers. The forecasting models applied are shown below:
- [Overview](#overview)
- [Holt-Winter](#holt-winter)
- [SARIMA model](#SARIMA-model)
- [Support Vector Regression](#SVR-model)
- [Hybrid SARIMA and SVR](#hybrid-SARIMA-vs-SVR)
- [Accuracy prediction](#performance-indicators)

I focus on the sale of Europe market given Europe has sold the greatest number of air fryer, accounting for 35.5% of total global volume of Philips air fryers.

# Overview
The modelling to forecast the quantity of air fryers is executed in Python. Different methods, including single one and hybrid one, are tested in order to find out which model has the best performance. Before the modelling, the first step is to look at the data plot of historical data, which is depicted in Figure 1: 

![image](https://user-images.githubusercontent.com/69800336/134210804-f1e58a23-ff9b-42fc-911c-c175e4e037e0.png)

As can be seen in Figure 1 1, the air fryer historical sales shows a strong seasonality and an increasing trend through years. Overall, throughout the given period, Philips air fryers sales data shows an upward trend interweaving with seasonal cycles that repeat within the year. The latest sales for 2021 are four times greater than the average sales of 2016. The author notices that the growth in sales for the last three months of the given years seem to be the highest compared to the figures for other months of the year. The three outstanding peaks of Philips air fryer took place in October and November of years of 2017, 2019, and 2020.

Moreover, in order to disprove the dummy assumptions of patterns perceived, the study conducts to decompose data set to check the trend or seasonality phenomenon. The decomposition of data splits data into trend and seasonal patterns which are shown in Figure 2 below:

![image](https://user-images.githubusercontent.com/69800336/134211059-9cd36ccc-9984-4c7f-893c-3392146b1658.png)

As observed in Figure 1 2, the data set was decomposed in four parts: observed, trend, seasonal, and residual. It is apparent that air fryer sales witnessed an upward trend and strong seasonality.

Preliminary analysis is executed to inspect features including the trend pattern, and the seasonality component, then the statistical method is applied to test whether the data is non-stationary. From the observed values, we can see that there is no clear hint of non-stationary within the time series as well as no clear trend. Thus non-seasonal differencing was not necessary. To confirm this the Augmented Dicky-Fuller (ADF) test was performed to test the null hypothesis of a unit root. With a ADF p-value less than 0.05 this resulted in not rejecting the null hypothesis of non-stationary. On the other hand, the trend values are currently being tested and their stationary characteristics are being verified by KPSS. Based upon the p-value of KPSS test smaller than 0.05, there is a clear evidence for rejecting the hypothesis of stationary series. In short, the data series is non-stationary.

![image](https://user-images.githubusercontent.com/69800336/134211410-b230764e-8ca2-48d9-925f-1845b7664f94.png)

Let’s plot the Partial Autocorrelation (PACF) and Autocorrelation (ACF) in the Figure 3 to test whether is a sign of heteroscedasticity. 

![image](https://user-images.githubusercontent.com/69800336/134211456-2c896995-d535-4574-8644-e0e240022ceb.png)
![image](https://user-images.githubusercontent.com/69800336/134211482-ed7b88c8-6605-427e-9fa6-277e52902576.png)

Looking at the plots of ACF and PACF, the orders indicate the possibility of seasonal components. The significant peak of both ACF and PACF plots is at lag 1, and have following peak at lag 12, then repeat the cycle of seasons. It suggests that both seasonal and non-seasonal autoregressive order could start with 1. Most of ACF and PACF residual values of seasonal orders are within significant interval at 95% confidence level. However, the existence of the ACF and PACF orders beyond the tolerant confidence limits suggests the presence of autocorrelation between residuals. 

# Holt-Winter
The seasonal and trend components of the data are not fixed in absolute numbers but are relatively constant variations in percentage terms around the underlying level of the data. The trend components will be owing to variables such as monthly production being impacted by population demand or market expansion, while the seasonal components will be due to the impacts of the season on sales. Thus, the study suggests that seasonal patterns and trend components are seen to interact in multiplicative manners to generate the finest predicted value in Holt-Winter model. The level constant is determined using a weighted average of the current seasonal component estimation and the preceding seasonal component estimate over a 12-period period. In the instance of Winter Exponential Smoothing, the random components affect the seasonal influence.

The level index shows the discrepancy from current data to future changes while the trend index offers an assessment of the movement shifting from time to time. The seasonality index evaluates the deviation in seasonality from a local mean. The coefficients of level terms, trend terms, and seasonal terms are 0.005; 0.005; and 0.639 respectively. Obviously, the smoothing parameters of level and trend patterns are relatively close to zero. In contrast, the coefficient of seasonal terms of over 0.6 indicates the seasonal component gives much more weight to the time-series data as opposed to the level and trend smoothing parameters. The Holt-Winter outcome emphasizes my expectation that the data of Philips air fryer is strongly dominated by the seasonality of orders. 

![image](https://user-images.githubusercontent.com/69800336/134211733-f27eaa7d-9a05-4e49-82db-73e7e5a158a1.png)

After calculating the smoothing constants for level, trend, and seasonality to optimise the errors of demand forecasting of Philips, the forecasting line is shown in the Figure 1 4. Given the application of exponential smoothing model considering with the components of trend, seasonality, and level, the Holt-Winter prediction model tends to move consistent with the real data, yet has a significant margin of error between forecasting and actual data.

![image](https://user-images.githubusercontent.com/69800336/134211784-0d0eadd4-5b74-4f84-83f3-f13d67ac64bd.png)

# SARIMA model
The seasonality of the 12 months presented in the given time series, the SARIMA method decomposes the components of trend, seasonality and residuals. The application of a 12-period moving average attempts to separate the seasonal effects from the time series, and then calculates the seasonal index. Looking at the trend index, it can be seen that Philips’ sales have steadily increased towards the end of the year. It is clear that the addition of seasonal, and trending components allows us to better predict overall long-term sales trends. To separate random from a trend after deseasonalisation, trend equations often use regression techniques based on the characteristic decreasing series in question and the time index.

Since the data is not a stationary time-series data, there is no constancy over times. In order to avoid the heteroscedasticity errors obtained from the model, the dataset needs to be rendered to a stationary series before using for the modelling. The typical way of converting a non-stationary into a stationary series is to subtract the previous observation from the current observation. However, with the support of function SARIMAX, this process can be simply done by adjusting the non-seasonal parameter d from 0 to 1. When the non-seasonal differencing d equals to 1, the data set is automatically converted as the order-1 differencing time series. The non-stationary of the observed data is identified with the non-seasonal differencing d is 1. Besides, the seasonal differencing parameter D is also equal to 1 in order to ensure the seasonal stationary of time-series data. Hence, SARIMA model (p, 1, q) x (P, 1, Q)12 is proposed. Drawing from the Table 1 3, the SARIMA(1, 1, 3) x (0, 1, 1)12 performs best with the lowest AIC value of 802.77. With the order of p of 1, the present time-series data y_t relies on the prior data y_(t-1). With the order of P of 0, the current year y_t is likely to not be reliant on its earlier year’s data. With the order of q of 3 and the order of Q of 1, the current time-series y_t is affected by the forgoing random shocks.

![image](https://user-images.githubusercontent.com/69800336/134212498-929a610a-69c0-4850-b567-c94ca90077d7.png)

The summary of SARIMA output is shown in Table 1 4. Based on p-value of parameters smaller than 0.05, all coefficients of SARIMA model are statistically significant. The p-value of Ljung-Box of 0.64 shows that there is no autocorrelation in the residuals, so they are independently distributed over time.

![image](https://user-images.githubusercontent.com/69800336/134212695-2c0573be-193d-47fb-abd7-b85d0093e8d8.png)

After estimating the non-seasonal parameters and seasonal auto-aggressive parameters in SARIMA model to optimise the errors of demand forecasting of Philips, the forecasting line is shown in the Figure 1 5. The SARIMA forecasting model tends to move in accordance with the movement of real data with smaller error margin as opposed to Holt-Winter model. The SARIMA forecasting methods is likely to give the most consistent performance among all methods.

![image](https://user-images.githubusercontent.com/69800336/134212782-20049b36-7c8d-4efe-a3b1-d43c6b24c79f.png)

# SVR model
The SVR is the linear model that can categorize data not statistically separated, and classify them into linear problem. SVR is a predictor of neural network based on the idea of learning algorithms and the structural risk reduction principle. SVR may simplify complicated non-linear regression issues in a high-dimensional function into linear regression issues.

Given the volatile series, the radial basis function kernel (RBF) or Gaussian Kernel is adopted rather than the linear default kernel function in SVR method. The study is performed using the cross-validation technique and RBF classification, with optimum parameter settings being achieved for SVR when the distance among points in the origin space is lowest. There is no standard measurement to identify the free parameters C, the gamma kernel function and the loss function ε for the SVR model. After running a cross validation technique, the value of free parameters C is varied from the range 0.001 to 100, while, and the gamma kernel function run from the range 0.0001 to 10. The search procedure with optimal parameters (C=1; gamma = 1) is performed best to cover the spread of data among generations. The accuracy of model is greater with the high value of C parameters and gamma parameter. The entire process of performance has been documented. 

After estimating the machine learning function and the free parameter in SVR model to optimise the errors of demand forecasting of Philips, the forecasting line is shown in the Figure 1 6. The SVR forecasting model mostly move consistent with the movement of real data, yet with larger error margin as opposed to Holt-Winter model. The orders following April 2021 of SVR forecasting model tend to go opposite to turning point with the actual direction of real data. Given the standard SVR algorithm adopts fixed values of regularization constant (Cao & Gu, 2002), SVR method is not likely to perform well when the non-stationary series has heterogeneous noises. 

![image](https://user-images.githubusercontent.com/69800336/134212856-c12212a2-c3d0-434e-8fbf-dc04136e5d95.png)

# Hybrid SARIMA vs SVR
The hybrid model used in the study integrates the seasonal auto-regressive integrated moving average method (SARIMA) and the support vector regression method (SVR). In order to estimate and analyse linear component of the air fryer sales, two steps are applied to the methodology used for determine the hybrid model: firstly a SARIMA model is developed, and then an SVM model is developed to determine non-linear patterns in a series of residual generated with the SARIMA model. In the hybrid model, SARIMA model is first applied to filter out the residual components contained in forecasting variables. The filtered forecasting variables are then used random forest approach in SVR for constructing a forecasting model based on permutation performance. 

After estimating parameters in the hybrid SARIMA-SVR model to optimise the errors of demand forecasting of Philips, the forecasting line is shown in the Figure 1 7. The SARIMA-SVR forecasting model tends to move in accordance with the movement of real data with smaller error margin. The performance of hybrid SARIMA-SVR model outperforms the benchmark model as Holt-Winter model.

![image](https://user-images.githubusercontent.com/69800336/134212913-73c1073a-f1cf-48f7-95b4-cb00bd2c1cfd.png)

# Performance indicators
Figure 1 8 shows the comparison of actual sales data, Philips LTSP forecast values, and prediction results of time-series models and machine learning models. We find that the forecast results of LTSP are much lower than the actual data. The first part of LTSP's forecast is quite close to the actual data, but the tail is calculated from the end of 2020, the data of LTSP is a bit flat and the tail is quite far from the real data. At the tail end of the time model, from February 2021, the actual sales data of Philips air fryers marks a sharp growth spurt after a period of sales decline. The SARIMA model produces the forecast results that seem to match the real data best, when the forecast data starts to bounce back from February 2021, but does not record a strong growth like the actual demand patterns. The Holt-Winter forecast model and the Philips LTSP model also predict the same direction, but the predicted data is too low compared to the real data. The prediction data of the Hybrid SARIMA-SVR model is not too far from the actual data, but the model often produces noisy forecast results at some moments when the prediction results deviate from the actual trend. The weird thing is that the SVR predictive model at the tail end is going down while the real line is going up. In Python code, after overfitting data with Random Forest, SVR prediction model is likely to be very resistant, and generates the poor performance. For SVR, an important attribute for successful training of the SVR model is the appearance of a constant relationship between the independent and dependent variables (Cao & Gu, 2002), however the Philips sales non-stationary time-series model dampens the accuracy of the prediction model of SVR.

![image](https://user-images.githubusercontent.com/69800336/134213285-7b60b86f-545d-4448-a7ad-e0e6f3fc2197.png)

The study assesses the forecasting performance in the sample, which the author utilizes to provide suggestions on how to use the sample prediction of our approach. The performance of the aforementioned time-series forecasting models, as well as the neural network machine learning algorithm methods, are measured based on the accuracy measurements. Among performance indicators, Root Mean Squared Error (RMSE) and Mean Absolute Percentage Errors (MAPE) are good approaches to assess the forecast accuracy. MAPE tells us how the average discrepancy between the predicted value and the actual. However, the measure could be misleading due to the abnormal value, hence RMSE provides the error measure from the perspective of standard deviation of data to give the accuracy of prediction. There is no standard of good MAPE and RMSE error measures to evaluate good forecast model. The forecast accuracy could be determined by comparing the performance of benchmark model, singular models, and hybrid one. In the study, Holt-Winters serves as a comparison benchmark towards SARIMA, SVR, and hybrid models.

Table 1 5 shows the performance of the models. Clearly, it is demonstrated that hybrid SARIMA-SVR model has also significant low RMSE value considering Europe market. Taking a closer look at the Table 1 5, the author sees the same pattern for MAPE value of hybrid SARIMA-SVR method. Surprisingly, in contrast to the literature review that neural networks and machine learning method have proven to have better accuracy, SVR model has the poorest forecasting performance among methods. That could be probably given the non-stationary series has noises and the standard SVR algorithm adopts fixed values of regularization constant. Thus, SVR performs badly in predict Philip air fryer sales. The integration of time-series analysis into SVR machine learning method could improve the performance of several aspects of non-stationary dataset in comparison with the accuracy of SVR model. Thus, it is not surprising that the hybrid model to lead to a better outcome. By contrast, the SARIMA method indeed deliver the best performance, based on both RMSE and MAPE. This is understandable because of the strong indication of the seasonal component yet non-stationary phenomenon in the series. Based on the concept of non-seasonal and seasonal auto aggressive moving average, SARIMA model is the best forecasting model among all methods to predict the long-term sale of Philips air fryers. With SARIMA model, it is expected that if the sale data of Philips air fryer has been performed with strong seasonal patterns in the long-term, the improvement with SARIMA could have been smoothed and achieved the closest prediction.

![image](https://user-images.githubusercontent.com/69800336/134213419-159e19d2-08d2-4f51-92fc-039d22d62e37.png)


### © 2021 linhvientran



