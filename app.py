from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)

# Function to progress the match
def match_progression(x_df, match_id):
    # Implement this function
    pass

@app.route('/')
def home():
    teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
        'Rajasthan Royals', 'Delhi Capitals'
    ]
    cities = [
        'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru'
    ]
    return render_template('index.html', teams=teams, cities=cities)


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)

    # Predict using the model
    result = pipe.predict_proba(pd.DataFrame(data))

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5500,debug=True)
