import numpy as np
import pandas as pd
import kagglehub
import os, time, warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

SEED = 42

DATA_DIR = os.environ.get('OLIST_DIR')
DATA_DIR = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

ler = lambda nome: pd.read_csv(os.path.join(DATA_DIR, nome))
orders = ler('olist_orders_dataset.csv')
items = ler('olist_order_items_dataset.csv')
customer = ler('olist_customers_dataset.csv')
seller = ler('olist_sellers_dataset.csv')
products = ler('olist_products_dataset.csv')
geo = ler('olist_geolocation_dataset.csv')
cat_trad = ler('product_category_name_translation.csv')

