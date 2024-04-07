from flask import Flask,render_template,request
from flask_cors import CORS
from flask import request, jsonify
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import load

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
  
    prob = LpProblem("Squad_Selection", LpMaximize)

  
    players = players_collection.find()  
    selected = {player["player"]: LpVariable(player["player"], 0, 1, LpBinary) for player in players}

   
    prob += sum(player["predicted_performance"] * selected[player["player"]] for player in players)

 
    prob += sum(selected[player["player"]] for player in players) >= 18  # Minimum squad size
    prob += sum(selected[player["player"]] for player in players) <= 25  # Maximum squad size

  
    prob += sum(selected[player["player"]] for player in players if player["Player_Type"] == 0) >= 6  # Batsman
    prob += sum(selected[player["player"]] for player in players if player["Player_Type"] == 1) >= 4  # Spinner
    prob += sum(selected[player["player"]] for player in players if player["Player_Type"] == 2) >= 3  # Pacer
    prob += sum(selected[player["player"]] for player in players if player["Player_Type"] == 3) >= 3  # WicketKeeper
    prob += sum(selected[player["player"]] for player in players if player["Player_Type"] == 4) >= 4  # Allrounder


    prob += sum(selected[player["player"]] for player in players if player["Nationality"] == 1) <= 8

  
    budget = 100 
    prob += sum(player["predicted_price"] * selected[player["player"]] for player in players) <= budget

   
    prob.solve()

    selected_players = [player for player in players if selected[player["player"]].varValue == 1]


    db.selected_squads.insert_many(selected_players)

    total_predicted_price = sum(player["predicted_price"] for player in selected_players)
    remaining_budget = budget - total_predicted_price

  
    selected_squad_response = [
        {
            "Player": player["player"],
            "Player Type": "Batsman" if player["Player_Type"] == 0 else "Spinner" if player["Player_Type"] == 1 else "Pacer" if player["Player_Type"] == 2 else "Wicketkeeper" if player["Player_Type"] == 3 else "Allrounder",
            "Nationality": "Overseas" if player["Nationality"] == 1 else "Indian",
            "Predicted Price (Crores)": f"{player['predicted_price']:.2f} Crores"
        } for player in selected_players
    ]


    selected_squad_response_sorted = sorted(selected_squad_response, key=lambda x: (x["Player Type"], x["Player"]))

    return jsonify({"selected_squad": selected_squad_response_sorted, "total_predicted_price": total_predicted_price, "remaining_budget": remaining_budget})


if __name__ == "__main__":
    app.run(debug=True,port=5500)
