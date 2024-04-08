from flask import Flask,render_template,request
from flask_cors import CORS
from flask import request, jsonify
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, LpSolverDefault
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from joblib import load
from waitress import serve

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb+srv://rohitsarje17:rohitsarje17@cluster0.kkcrswd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["cricket_db"]
players_collection = db["players"]


price_model = RandomForestRegressor(n_estimators=100, random_state=42)
performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model = load("price_model.pkl")
performance_model = load("performance_model.pkl")


Player_Type_mapping = {"Batsman": 0, "Spinner": 1, "Pacer": 2, "Wicketkeeper": 3, "Allrounder": 4}
Nationality_mapping = {"Indian": 0, "Overseas": 1}

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/players")
def show_players():
    return render_template('players.html')

@app.route("/get_all_players")
def get_all_players():
    all_players = list(players_collection.find({}, {'_id': 0}))
    return jsonify(all_players)


@app.route("/add_player.html",methods=["GET"])
def player():
    return render_template('add_player.html')

@app.route("/add_player", methods=["POST"])
def add_player():
    data = request.json
 
    Player_Type = Player_Type_mapping.get(data["Player_Type"], -1)
    Nationality = Nationality_mapping.get(data["Nationality"], -1)
    
    features = np.array([
        Player_Type, Nationality, data["runs"], data["boundaries"], data["balls_faced"],
        data["wickets"], data["balls_bowled"], data["runs_conceded"], data["matches"],
        data["batting_avg"], data["batting_strike_rate"], data["boundaries_percent"],
        data["bowling_economy"], data["bowling_avg"], data["bowling_strike_rate"],
        data["catches"], data["stumpings"]
    ]).reshape(1, -1)
    
    predicted_price = price_model.predict(features)[0]
    predicted_performance = performance_model.predict(features)[0]

    new_player = {
        "player":data["player"],
        "Player_Type": Player_Type,
        "Nationality": Nationality,
        "runs": data["runs"],
        "boundaries": data["boundaries"],
        "balls_faced": data["balls_faced"],
        "wickets": data["wickets"],
        "balls_bowled": data["balls_bowled"],
        "runs_conceded": data["runs_conceded"],
        "matches": data["matches"],
        "batting_avg": data["batting_avg"],
        "batting_strike_rate": data["batting_strike_rate"],
        "boundaries_percent": data["boundaries_percent"],
        "bowling_economy": data["bowling_economy"],
        "bowling_avg": data["bowling_avg"],
        "bowling_strike_rate": data["bowling_strike_rate"],
        "catches": data["catches"],
        "stumpings": data["stumpings"],
        "predicted_price": predicted_price,
        "predicted_performance": predicted_performance
    }

    players_collection.insert_one(new_player)
    
    return jsonify({"message": "Player added successfully"}), 201

@app.route("/generate_team", methods=["GET"])
def generate_team():
    player_data = list(players_collection.find({}, {'_id': 0}))

    df = pd.DataFrame(player_data)

    selected_squad, total_predicted_price, remaining_budget = select_team(df)

    print("Selected Squad:")
    for player_info in selected_squad:
        print(player_info)

    response_data = {
        "selected_squad": selected_squad,
        "total_predicted_price": total_predicted_price,
        "remaining_budget": remaining_budget
    }

    return jsonify(response_data)

def select_team(df):
    
    prob = LpProblem("Squad_Selection", LpMaximize)

 
    players = df['player'].tolist() 
    selected = LpVariable.dicts("Selected", players, 0, 1, LpBinary)  

  
    prob += sum(df.loc[df['player'] == player, 'predicted_performance'].iloc[0] * selected[player] for player in players)

    prob += sum(selected[player] for player in players) >= 18  
    prob += sum(selected[player] for player in players) <= 25  

   
    prob += sum(selected[player] for player in players if df.loc[df['player'] == player, 'Player_Type'].iloc[0] == 0) >= 6 
    prob += sum(selected[player] for player in players if df.loc[df['player'] == player, 'Player_Type'].iloc[0] == 1) >= 4  
    prob += sum(selected[player] for player in players if df.loc[df['player'] == player, 'Player_Type'].iloc[0] == 2) >= 3  
    prob += sum(selected[player] for player in players if df.loc[df['player'] == player, 'Player_Type'].iloc[0] == 3) >= 3  
    prob += sum(selected[player] for player in players if df.loc[df['player'] == player, 'Player_Type'].iloc[0] == 4) >= 4  

 
    prob += sum(selected[player] for player in players if df.loc[df['player'] == player, 'Nationality'].iloc[0] == 1) <= 8

 
    budget = 100 
    prob += sum(df.loc[df['player'] == player, 'predicted_price'].iloc[0] * selected[player] for player in players) <= budget

  
    prob.solve()

    selected_players = [player for player in players if selected[player].varValue == 1]
    total_predicted_price = sum(df.loc[df['player'] == player, 'predicted_price'].iloc[0] for player in selected_players)
    remaining_budget = budget - total_predicted_price

    selected_squad_data = []
    for player in selected_players:
        player_type = df.loc[df['player'] == player, 'Player_Type'].iloc[0]
        if player_type == 0:
            player_type_str = 'Batsman'
        elif player_type == 1:
            player_type_str = 'Spinner'
        elif player_type == 2:
            player_type_str = 'Pacer'
        elif player_type == 3:
            player_type_str = 'Wicket keeper'
        elif player_type == 4:
            player_type_str = 'Allrounder'
        player_price = df.loc[df['player'] == player, 'predicted_price'].iloc[0]
        player_nationality = 'Overseas' if df.loc[df['player'] == player, 'Nationality'].iloc[0] == 1 else 'Indian'
        selected_squad_data.append([player, player_type_str, player_nationality, f"{player_price:.2f} Crores"])


    headers = ["Player", "Player Type", "Nationality", "Predicted Price (Crores)"]

    selected_squad_data_sorted = sorted(selected_squad_data, key=lambda x: (x[1] != 'Batsman', x[1] != 'Allrounder', x[1] != 'Wicket keeper', x[1] != 'Spinner', x[1] != 'Pacer', x[0]))

    return selected_squad_data_sorted, total_predicted_price, remaining_budget

# if __name__ == "__main__":
#     app.run(debug=True,port=5500)

serve(app, host='0.0.0.0', port=5500)
