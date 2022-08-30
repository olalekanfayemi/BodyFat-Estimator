from flask import Flask, request, render_template


import pickle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


file1 = open('bodyfatmodel1.pkl', 'rb')
sv = pickle.load(file1)
file1.close()


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        my_dict = request.form
        print(my_dict)

        density = float(my_dict['density'])
        abdomen = float(my_dict['abdomen'])
        chest = float(my_dict['chest'])
        weight = float(my_dict['weight'])
        height = float(my_dict['height'])
        hip = float(my_dict['hip'])

        bmi = 703*weight/(height**2)
        caratio = abdomen/chest

        input_features = [[density, abdomen, chest, hip, weight, bmi, caratio]]
        in_features = sc.fit_transform(input_features)
        prediction = sv.predict(in_features)[0].round(2)

        # <p class="big-font">Hello World !!</p>', unsafe_allow_html=True

        string = 'Percentage of Body Fat Estimated is : ' + str(prediction)+'%'

        return render_template('show.html', string=string)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
