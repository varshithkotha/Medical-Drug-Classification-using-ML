from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('drug.pkl', 'rb'))

# Define the feature names used during training
feature_names = ['Sex_F', 'Sex_M', 'BP_HIGH', 'BP_LOW', 'BP_NORMAL', 'Cholesterol_HIGH', 'Cholesterol_NORMAL', 'Age_binned_<20s', 'Age_binned_20s', 'Age_binned_30s', 'Age_binned_40s', 'Age_binned_50s', 'Age_binned_60s', 'Age_binned_>60s', 'Na_to_K_binned_<10', 'Na_to_K_binned_10-20', 'Na_to_K_binned_20-30', 'Na_to_K_binned_>30']

def detect_drug(age, sex, bp, cholesterol, na_to_k):
    # Create a DataFrame with input data
    input_data = {
        'Sex': [sex],
        'BP': [bp],
        'Cholesterol': [cholesterol],
        'Age_binned': [age],  # Assuming age is binned
        'Na_to_K_binned': [na_to_k]  # Assuming Na_to_K is binned
    }
    input_df = pd.DataFrame(input_data)
    
    # Perform one-hot encoding for categorical variables
    input_df = pd.get_dummies(input_df)
    
    # Make sure input features match the columns used during training
    # Use the same order of columns as used during training
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Make predictions
    predicted_drug = model.predict(input_df)
    
    return predicted_drug[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        age = request.form['age']
        sex = request.form['sex']
        bp = request.form['bp']
        cholesterol = request.form['cholesterol']
        na_to_k = request.form['na_to_k']

        # Call detect_drug function
        predicted_drug = detect_drug(age, sex, bp, cholesterol, na_to_k)

        return render_template('index.html', prediction=predicted_drug)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
