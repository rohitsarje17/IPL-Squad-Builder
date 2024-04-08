import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
 

df = pd.read_csv('Datasets/IPLPlayersData.csv')

df['SellingPrice'] = df['SellingPrice'] / 10000000
df['BasePrice'] = df['BasePrice'] / 10000000

player_type_mapping = {
    "Batsman": 0,
    "Spinner": 1,
    "Pacer": 2,
    "wicketkeeper": 3,
    "Allrounder":4
}

nationality_mapping = {
    "Indian": 0,
    "Overseas": 1
}

df['Player_Type'] = df['Player_Type'].map(player_type_mapping)
df['Nationality'] = df['Nationality'].map(nationality_mapping)


features = df.drop(columns=['player', 'BasePrice', 'SellingPrice', 'PerformanceIndex'])
target_price = df['SellingPrice']
target_performance_index = df['PerformanceIndex']

X = features
y_price = target_price
y_performance_index = target_performance_index

print(features.head())

X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.3, random_state=42)

price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X_train_price, y_train_price)

print(price_model.score(X_train_price, y_train_price))

y_pred_price = price_model.predict(X_test_price)

mse_price = mean_squared_error(y_test_price, y_pred_price)
print("Mean Squared Error for Price Prediction:", mse_price)


# pickle.dump(price_model, open('price_model.pkl', 'wb'))

# price_model = pickle.load(open('price_model.pkl', 'rb'))

################################################################


X_train_performance, X_test_performance, y_train_performance, y_test_performance = train_test_split(X, y_performance_index, test_size=0.3, random_state=42)

performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
performance_model.fit(X_train_performance, y_train_performance)

y_pred_performance = performance_model.predict(X_test_performance)

mse_performance = mean_squared_error(y_test_performance, y_pred_performance)
print("Mean Squared Error for Performance Index Prediction:", mse_performance)

# pickle.dump(performance_model, open('performance_model.pkl', 'wb'))

# perfomance_model = pickle.load(open('performance_model.pkl', 'rb'))\


from joblib import dump

# Save the trained models
dump(price_model, "price_model.pkl")
dump(performance_model, "performance_model.pkl")

################################################################

virat = {
    'Player_Type': [0],  # Batsman
    'Nationality': [0],   # Indian
    'runs': [7579],
    'boundaries': [918],
    'balls_faced': [5802],
    'wickets': [4],
    'balls_bowled': [251],
    'runs_conceded': [368],
    'matches': [242],
    'batting_avg': [38],
    'batting_strike_rate': [130.6],
    'boundaries_percent': [55],
    'bowling_economy': [8.79],
    'bowling_avg': [92],
    'bowling_strike_rate': [62],
    'catches': [110],
    'stumpings': [0]
}

# rohit = {
#     'Player_Type': [0],  # Batsman
#     'Nationality': [0],   # Indian
#     'runs': [6280],
#     'boundaries': [823],
#     'balls_faced': [4818],
#     'wickets': [15],
#     'balls_bowled': [339],
#     'runs_conceded': [453],
#     'matches': [246],
#     'batting_avg': [29],
#     'batting_strike_rate': [130.3],
#     'boundaries_percent': [60],
#     'bowling_economy': [8],
#     'bowling_avg': [30],
#     'bowling_strike_rate': [22],
#     'catches': [99],
#     'stumpings': [0]
# }

# virat_df = pd.DataFrame(virat)
# rohit_df = pd.DataFrame(rohit)

# # Make prediction
# virat_price= price_model.predict(virat_df)
# virat_performance=perfomance_model.predict(virat_df)

# print("Virat Price", virat_price)
# print("Virat Performance", virat_performance)



# # Make prediction
# rohit_price= price_model.predict(rohit_df)
# rohit_performance=perfomance_model.predict(rohit_df)

# print("Rohit Price", rohit_price)
# print("Rohit Performance", rohit_performance)









