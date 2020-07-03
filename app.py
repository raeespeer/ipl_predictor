import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('ipl_predictor.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    stads = {'Rajiv Gandhi International Stadium, Uppal': 0, 'Maharashtra Cricket Association Stadium': 1, 'Saurashtra Cricket Association Stadium': 2, 'Holkar Cricket Stadium': 3, 'M Chinnaswamy Stadium': 4, 'Wankhede Stadium': 5, 'Eden Gardens': 6, 'Feroz Shah Kotla': 7, 'Punjab Cricket Association IS Bindra Stadium, Mohali': 8, 'Green Park': 9, 'Punjab Cricket Association Stadium, Mohali': 10, 'Sawai Mansingh Stadium': 11, 'MA Chidambaram Stadium, Chepauk': 12, 'Dr DY Patil Sports Academy': 13, 'Newlands': 14, "St George's Park": 15, 'Kingsmead': 16, 'SuperSport Park': 17, 'Buffalo Park': 18, 'New Wanderers Stadium': 19, 'De Beers Diamond Oval': 20, 'OUTsurance Oval': 21,
             'Brabourne Stadium': 22, 'Sardar Patel Stadium, Motera': 23, 'Barabati Stadium': 24, 'Vidarbha Cricket Association Stadium, Jamtha': 25, 'Himachal Pradesh Cricket Association Stadium': 26, 'Nehru Stadium': 27, 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 28, 'Subrata Roy Sahara Stadium': 29, 'Shaheed Veer Narayan Singh International Stadium': 30, 'JSCA International Stadium Complex': 31, 'Sheikh Zayed Stadium': 32, 'Sharjah Cricket Stadium': 33, 'Dubai International Cricket Stadium': 34, 'M. A. Chidambaram Stadium': 35, 'Feroz Shah Kotla Ground': 36, 'M. Chinnaswamy Stadium': 37, 'Rajiv Gandhi Intl. Cricket Stadium': 38, 'IS Bindra Stadium': 39, 'ACA-VDCA Stadium': 40}
    toss = {'field': 0, 'bat': 1}
    teams = {'MI': 1,
             'KKR': 2,
             'RCB': 3,
             'DC': 4,
             'CSK': 5,
             'RR': 6,
             'DCS': 7,
             'GL': 8,
             'KXIP': 9,
             'SRH': 10,
             'RPS': 11,
             'KTK': 12,
             'PW': 13,
             'tie': 14,
             }

    team1 = request.form['tem1']
    team2 = request.form['tem2']
    toss_winner = request.form['win']
    toss_decision = request.form['toss']
    venue = request.form['venue']
    int_features = [teams[team1],teams[team2],teams[toss_winner],toss[toss_decision],stads[venue]]

    # print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    outs = list(teams.keys())[list(teams.values()).index(output)]

    return render_template('index.html', prediction_text='The winner will be {}'.format(outs))


if __name__ == "__main__":
    app.run(debug=True)
