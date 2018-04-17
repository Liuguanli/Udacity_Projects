# Machine Learning Engineer Nanodegree
## Capstone Proposal
GuanLi Liu
April 16th 2018

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

Forecast Rossmann Store Sales Project is a Kaggle competition which forecasts sales using store, promotion, and competitor data. In statistics, prediction is a part of statistical inference. One particular approach to such inference is known as predictive inference, but the prediction can be undertaken within any of the several approaches to statistical inference.When information is transferred across time, often to specific points in time, the process is known as forecasting.

It is obviously that this is a time series related problem. A time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. 【Wiki】
Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values. So, in this problem we should analysis the dataset first and then forecast the result.

Time series forecasting is essential in our daily life like the temperature forecast is quite convenient for our life. The forecast of stock price is helpful for some business man to earn more money. Also the selling forecast can help the merchants to cut cost.

### Problem Statement
_(approx. 1 paragraph)_

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. The goal of the Project is to predict 6 weeks of daily Sales in 1115 stores located in different parts of Germany based on 2.5 years of historical daily sales. I suppose the problem can be solved by using Random Forest.

The training algorithm for random forests applies the general technique of bootstrap aggregating, or bagging, to tree learners. Given a training set $X = x_{1}, ..., x_{n}$ with responses $Y = y_{1}, ..., y_{n}$, bagging repeatedly selects a random sample with replacement of the training set and fits trees to these samples: For $b = 1, ..., B$ : Sample, with replacement, n training examples from $X$, $Y$; call these $X_{b}$, $Y_{b}$. Train a classification or regression tree $f_{b}$ on $X_{b}$, $Y_{b}$. After training, predictions for unseen samples $x'$ can be made by averaging the predictions from all the individual regression trees on $x'$: $ {\displaystyle {\hat {f}}={\frac {1}{B}}\sum _{b=1}^{B}f_{b}(x')} $. Submissions are evaluated on the Root Mean Square Percentage Error (RMSPE) which is introduced in the official website. The RMSPE is calculated as $ RMSPE=\sqrt{\frac{1}{n}\cdot \sum_{i=1}^{n}\left ( \frac{^{y_{i} - \hat{y_{i}}}}{y_{i}} \right )^{2}} $.  where $y_{i}$ denotes the sales of a single store on a single day and $\hat{y_{i}}$ denotes the corresponding prediction. Any day and store with 0 sales is ignored in scoring.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

There are four data Files given by Kaggle including:
> * train.csv - historical data including Sales
> * test.csv - historical data excluding Sales
> * sample_submission.csv - a sample submission file in the correct format
> * store.csv - supplemental information about the stores

#### Data fields:
Most of the fields are self-explanatory like DayOfWeek and Date. While the following are descriptions for those that aren't.

> * Id - an Id that represents a (Store, Date) duple within the test set
> * Store - a unique Id for each store
> * Sales - the turnover for any given day (this is what you are predicting)
> * Customers - the number of customers on a given day
> * Open - an indicator for whether the store was open: 0 = closed, 1 = open
> * StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
> * SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
> * StoreType - differentiates between 4 different store models: a, b, c, d
> * Assortment - describes an assortment level: a = basic, b = extra, c = extended
> * CompetitionDistance - distance in meters to the nearest competitor store
> * CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
> * Promo - indicates whether a store is running a promo on that day
> * Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
> * Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
> * PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

All of the datasets should be used and we can choose some essential information from them.Customers and Open is of vital importance. Because they are positively correlated with sales. Besides, the imformation about promotion is also useful because people tend to buy more goods. The type, Competition Distance, holiday also have some influence on the sale. So, in the following time, I will try those given information to predict the sale.

However, there are some flaws in the dataset. For example, as I watched in the forum that Store 622 has 11 missing values in the Open columns and most people assume the store is open. If it's closed then sales will be 0 and it won't count toward the score. But if it's open and you predict 0 then you're in for some heavy penalty. Additionally, there is a 6 month gap where we have a smaller number of stores reporting sales revenue. We also have to impute the missing values to have a stable result. Finally, we have to deal with the value of some fields which are not suitable for the train model. Like: StoreType and Assortment. The values of them are letters we should change letter to valus like 1-4 for StoreType and 1-3 for Assortment respectively.

### Solution Statement
_(approx. 1 paragraph)_

给出一个问题的解决方案，

Entity Embedding；神经网络

---

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

https://www.kaggle.com/shearerp/store-dayofweek-promo-0-13952/code
There were two simple benchmark models (median and geometric mean) on the competition forum which I used as a starting point.

---

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

整个工作的流程

如何分析数据

可以使用什么方法，我应该使用什么方法

图，伪代码，可视化 来描述项目的设计。


In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
