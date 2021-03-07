import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMRegressor
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from helpers.data_prep import *
from helpers.eda import *
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# data
data = pd.read_csv("datasets/hitters.csv")
df = data.copy()
df.head()

df.isnull().sum()
df.dropna(inplace=True)
df.shape

################################
# HITTERS
################################

f"""

AtBat: Number of times at bat in 1986

Hits: Number of hits in 1986

HmRun: Number of home runs in 1986

Runs: Number of runs in 1986

RBI: Number of runs batted in in 1986

Walks: Number of walks in 1986

Years: Number of years in the major leagues

CAtBat: Number of times at bat during his career

CHits: Number of hits during his career

CHmRun: Number of home runs during his career

CRuns: Number of runs during his career

CRBI: Number of runs batted in during his career

CWalks: Number of walks during his career

League: A factor with levels A and N indicating player's league at the end of 1986

Division: A factor with levels E and W indicating player's division at the end of 1986

PutOuts: Number of put outs in 1986

Assists: Number of assists in 1986

Errors: Number of errors in 1986

Salary: 1987 annual salary on opening day in thousands of dollars

NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987

"""

df.corr(method="spearman")
plt.figure(figsize=(15, 7))
sns.heatmap(df.corr(method="spearman"), annot=True)
plt.show()

# outliers

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

for col in num_cols:  # num kolonlarda thresholds
    print(col, outlier_thresholds(df, col))
for col in num_cols:  # num kolonlarda outlier kontrolu
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)


df.head()

df['AtBat_ratio'] = df['AtBat'] / df['CAtBat']
df['Hits_ratio'] = df['Hits'] / df['CHits']
df['HmRun_ratio'] = df['HmRun'] / df['CHmRun']
df['Runs_ratio'] = df['Runs'] / df['CRuns']
df['RBI_ratio'] = df['RBI'] / df['CRBI']
df['Walks_ratio'] = df['Walks'] / df['CWalks']


df.dtypes
binary_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

ohe_cols = [col for col in df.columns if df[col].dtypes == 'O' and df[col].nunique() > 2]

for col in ohe_cols:
    df = one_hot_encoder(df, ohe_cols, True)


y = df["Salary"]
X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')




#######################################
# LightGBM: Model & Forecast
#######################################

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#######################################
# Feature Importance
#######################################
plot_importance(lgbm_tuned, X_train)


