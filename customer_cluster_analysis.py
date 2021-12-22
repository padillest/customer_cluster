"""
The purpose of this analysis was to identify 
clusters within a customer base through the 
use of Principal Component Analysis (PCA) 
and k-Means Clustering. 
""" 

# Importing libraries 

import numpy as np
import pandas as pd 
from pandas.api.types import CategoricalDtype 

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer

# Loading data 

data = pd.read_csv(
    filepath_or_buffer = "marketing_campaign.csv",
    sep = "\t")

df = data.copy()


# Creating functions

def data_check(data: pd.DataFrame):
    """Returns information of the current dataset

    This function takes a pandas dataset and produces
    the first three entries, a summary of variables, 
    a list of columns, and checks for missing values.
    """
    display(data.head(3))
    display(data.describe())
    display(data.info())
    missing_value_df = pd.DataFrame(df.isnull().sum(),
                                    columns = ["missing_values"])
    if missing_value_df["missing_values"].sum() > 0:
        missing_value_df.plot(kind="bar")
    else:
        print("There is no missing data")

def boxplot_grid(df: pd.DataFrame, col_list: list,
                 rows: int, cols: int, size=(20,16)):
    """Returns boxplots of the columns of a dataset

    This function will create a grid of boxplots of 
    the columns within the given dataframe and list.
    """
    fig, axs = plt.subplots(rows, cols, figsize=size)
    axs = axs.flatten()
    for i, data_item in enumerate(col_list):
        sns.boxplot(data = df[data_item],
                    ax = axs[i])
        axs[i].set_title(data_item)

def histogram_grid(df: pd.DataFrame, col_list: list,
                   rows: int, cols: int, size=(20,16)):
    """Returns histograms of the columns of a dataset

    This function will create a grid of histograms of
    the columns within the given dataframe and list.
    """
    fig, axs = plt.subplots(rows, cols, figsize=size)
    axs = axs.flatten()
    for i, data_item in enumerate(col_list):
        mean, median = df[data_item].mean(), df[data_item].median()
        graph = sns.histplot(df[data_item], 
                             ax=axs[i])
        mean_line = graph.axvline(mean,
                                  c="red",
                                  label="mean")
        median_line = graph.axvline(median,
                                    c="green",
                                    label="median")
        plt.legend(handles=[mean_line, median_line],
                   labels=["mean", "median"])
        axs[i].set_title(data 
                         + ", skew: "
                         + str(round(df[data_item].skew(axis=0))))

def bar_grid(df: pd.DataFrame, col_list: list,
             x_var="education", hue_var="marital_status",
             rows: int, cols: int, size=(20,16)):
    """Returns bar plots in relation to two categorical variables

    This function will create a grid of bar plots of the given 
    columns of a specified dataframe.
    """
    fig, axs = plt.subplots(rows, col, figsize=size)
    axs = axs.flatten()
    for i, data_item in enumerate(col_list):
        graph = sns.barplot(ax=axs[i],
                            data=df,
                            x=x_var,
                            y=col_list[i],
                            hue=hue_var)
        axs[i].set_title(data_item)

def dis_grid(df: pd.DataFrane, col_list: list):
    """Returns a grid of distribution plots

    This function will create a grid of distribution
    plots from a given list of columns within a dataframe.
    """
    for i, data_item in enumerate(col_list):
        graph = sns.displot(data=df,
                            x=item,
                            hue="cluster",
                            col="cluster")
        axs[i].set_title(data_item)

# Viewing the data 

data_check(data=df)

# Data preprocessing

df.columns = df.columns.str.lower()

## Encoding variables 

### Education variable

#### Creating a dictionary of values to be changed 
edu_replace = {
    "Graduation":"Graduate",
    "Master":"Postgraduate",
    "PhD":"Postgraduate",
    "2n Cycle":"Undergraduate",
    "Basic":"Undergraduate"
}

#### Replacing values and ordering levels 
df = df.assign(
    education = df["education"].replace(
        edu_replace
    ).astype(
        CategoricalDtype(
            categories = [
                "Undergraduate",
                "Graduate",
                "Postgraduate"
            ],
            ordered=True
        )
    )
)

### Marital status variable 

#### Creating a dictionary of values to be changed 
marital_replace = {
    "Married":"Relationship",
    "Together":"Relationship",
    "Divorced":"Single",
    "Widow":"Single",
    "Alone":"Single",
    "YOLO":"Single",
    "Absurd":"Single"
}

#### Replacing values and ordering levels 
df = df.assign(
    marital_status = df["marital_status"].replace(
        marital_replace
    ).astype(
        CategoricalDtype(
            categories = [
                "Single",
                "Relationship"
            ],
            ordered=True
        )
    )
)

## Feature engineering

### Age 

#### Generating current age
age = 2021-df["year_birth"]

#### Creating variable with current age
df = df.assign(age = age)

### Number of people home 

#### Creating a variable with a sum of youth in a household
df = df.assign(num_child = df["kidhome"]+df["teenhome"])

### Customer membership

#### Converting variable to datetime type 
df = df.assign(
    dt_customer = pd.to_datetime(
        df["dt_customer"]
    )
)

#### Creating a variable based on the newest and oldest member length
df = df.assign(
    member_length = max(df["dt_customer"]-df["dt_customer"]).dt.days)
)

### Total spending

#### Creating a total amount spent variable
df = df.assign(
    total_spent = df["mntwines"]
                  + df["mntfruits"]
                  + df["mntmeatproducts"]
                  + df["mntfishproducts"]
                  + df["mntsweetproducts"]
                  + df["mntgoldprods"]
)

### Number of purchases 

#### Creating a total number of purchases variable
df = df.assign(
    total_purchase = df["numdealspurchases"] 
                     + df["numwebpurchases"] 
                     + df["numcatalogpurchases"] 
                     + df["numstorepurchases"]
)

## Dropping columns 

drop_col = [
    "id", "year_birth", "dt_customer",
    "z_costcontact", "z_revenue", "complain",
    "response"
]

df.drop(drop_col, axis=1, inplace=True)

## Handling missing values 

#### Creating heatmap of variables and their 
#### associated correlations with income
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(),
            annot=True,
            fmt=".1")

#### Filling missing values with the average 
#### income of entries grouped by education 
#### and the number of kids home
df["income"] = df["income"].fillna(
    df.groupby(
        ["education", "kidhome"]
    )
    ["income"].transform(
        "mean"
    )
)

## Outlier detection 

#### Creating list of numeric columns 
num_col = [
    "income", "recency", "mntwines", 
    "mntfruits", "mntmeatproducts", "mntfishproducts", 
    "mntsweetproducts", "mntgoldprods", "numdealspurchases", 
    "numwebpurchases", "numcatalogpurchases", "numstorepurchases", 
    "numwebvisitsmonth", "age", "member_length"
]

#### Producing a boxplot of columns 
boxplot_grid(
    df=df,
    col_list=num_col,
    rows=3,
    cols=5
)

#### Filtering outliers beyond our given IQR 
filter_col = [
    "income", "mntwines", "mntfruits",
    "mntmeatproducts", "mntfishproducts", "mntsweetproducts",
    "mntgoldprods", "numdealspurchases", "numwebpurchases",
    "numcatalogpurchases", "numwebvisitsmonth", "age"
]

#### Creating quantile cutoffs 
q1 = df[filter_col].quantile(0.0)
q3 = df[filter_col].quantile(0.85)

#### Calculating IQR
iqr = q3 - q1

#### Filtering the data based on quartiles and IQR 
df_filtered = df[~(
    (df[filter_col] < (q1 - 1.5*iqr)) | (df[filter_col] > (q3 - 1.5*iqr))
).any(axis=1)
]

# Data exploration

## Univariate analyses

### Demographics

#### Creating a list of demographic variables
demo_col = [
    "income", 
    "kidhome",
    "teenhome", 
    "age",
    "recency",
    "member_length"
]

#### Creating a grid of histograms 
histogram_grid(
    df=df_filtered,
    col_list=demo_col,
    rows=2,
    cols=3
)

### Product purchases

#### Creating a list of product-related columns
product_col = [
    x for x in df_filtered.columns.tolist() if x.startswith("mnt")
]

#### Creating a grid of histograms
histogram_grid(
    df=df_filtered,
    col_list=product_col,
    rows=2,
    cols=3
)

### Means of purchases

#### Creating a list of purchase-related columns 
purchase_col = [
    x for x in df_filtered.columns.tolist() if x.startswith("num") 
    and x != "num_child"
    and x != "numwebvisitsmonth" 
    and x != "numdealspurchases" 
    or x == "total_purchase"
]

#### Creating a grid of histograms
histogram_grid(
    df=df_filtered,
    col_list=purchase_col,
    rows=2,
    cols=3
)

## Categorical analyses

### Categorical variables and product purchases

#### Creating a grid of bar graphs
bar_grid(
    df=df_filtered, 
    col_list=product_col,
    rows=2,
    cols=3,
    x_var="education",
    hue_var="marital_status"
)

### Categorical variables and purchase points

#### Creating a grid of bar graphs
bar_grid(
    df=df_filtered, 
    col_list=purchases_col,
    rows=2,
    cols=3,
    x_var="education",
    hue_var="marital_status"
)

### Categorical variables and promotions

#### Creating a list of promotion-related variables
promo_col = [
    x for x in df_filtered.columns.tolist() if x.startswith("accepted") 
    or x == "numdealspurchases"
]

#### Creating a grid of bar graphs
bar_grid(
    df=df_filtered, 
    col_list=promo_col,
    rows=2,
    cols=3,
    x_var="education",
    hue_var="marital_status"
)

# Clustering

#### Creating a copy of the filtered data 
df_cluster = df_filtered.copy()

#### Removing binary columns
drop_col = [
    x for x in df_cluster.columns.tolist() if x.startswith("accepted")
]
df_cluster = df_cluster.drop(drop_col, axis=1)

## Encoding variables 

#### Creating a Label Encoder 
le = LabelEncoder()

#### Creating a list of categorical variables 
cat_var = [
    "education",
    "marital_status"
]

#### Iterating through categorical variables 
#### and applying label encoding to each column
for i, col in enumerate(cat_var):
    df_cluster[cat_var[i]] = le.fit_transform(
        df_cluster[cat_var[i]]
    )

for i, col in enumerate(cat_var):
    df_cluster[cat_var[i]] = le.fit_transform(
        df_cluster[cat_var[i]]
    )

## Standardizing all columns 

#### Creating a scaled dataset 
scaler = StandardScaler()
df_scale = pd.DataFrame(scaler.fit_transform(df_cluster),
                        columns=df.cluster.columns)

## Principal Component Analysis

#### Creating a dataset of three main variables 
pca = PCA(n_components=3)
pca_col = [
    "v1",
    "v2",
    "v3"
]
df_pca = pd.DataFrame(pca.fit_transform(df_scale),
                      columns=pca_col)

#### Using K-Elbow Visualizer to determine the ideal number of clusters
model = KMeans()
vis = KElbowVisualizer(model,
                       k=(2,8))
vis.fit(df_pca)
vis.show()

#### Applying clusters to the dataset
cluster = KMeans(n_cluster=4)
cluster_fit = cluster.fit_predict(df_pca)
df_filtered = df_filtered.assign(cluster=cluster_fit)

#### Visualizing the data through the predicted clusters
sns.scatterplot(data=df_filtered,
                x="total_spent",
                y="income",
                hue="cluster")

#### Demographics and clusters
dis_grid(df_filtered,
         demo_col)

#### Product purhcases and clusters
dis_grid(df_filtered,
         product_col)

#### Purchase points and clusters
dis_grid(df_filtered,
         purchase_col)

#### Promotion response and clusters
dis_grid(df_filtered,
         promo_col)

