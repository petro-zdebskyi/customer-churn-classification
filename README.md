# Customer churn classification
Small script for predicting customer churn.

Structure:
* `main.py` - has all the implementation.
* `data/input` - has input dataframe used for training and testing.
* `data/output` - resulting files (new dataframe, evaluation json, and model dump).

To enrich dataset monthly data of [US Consumer Price Index (CPI)](https://www.kaggle.com/datasets/varpit94/us-inflation-data-updated-till-may-2021) was used under the assumption that transactions are in USD or in currency related to it. Incorporating CPI allows to account for inflation/deflation and determine the real purchasing power of the amounts at different points in time.