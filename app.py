from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Configure Matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load model
try:
    model = pickle.load(open('car_price_model.pkl', 'rb'))
    columns = pickle.load(open("columns.pkl", "rb"))
except FileNotFoundError:
    print("Warning: Model or columns files not found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        year = int(request.form['year'])
        km_driven = float(request.form['km_driven'])
        fuel = int(request.form['fuel'])
        seller_type = int(request.form['seller_type'])
        transmission = int(request.form['transmission'])
        owner = int(request.form['owner'])
        mileage = float(request.form['mileage'])
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = int(request.form['seats'])

        # Create dataframe with all model columns
        input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        # Map inputs to DataFrame columns (using common column names as fallback)
        mapping = {
            'year': year, 'km_driven': km_driven, 'owner': owner, 
            'mileage(km/ltr/kg)': mileage, 'engine': engine, 
            'max_power': max_power, 'seats': seats,
            'fuel': fuel, 'seller_type': seller_type, 'transmission': transmission
        }
        
        for col, val in mapping.items():
            if col in input_data.columns:
                input_data[col] = val

        prediction = model.predict(input_data)[0]

        # Visualization: Generate a performance plot
        plt.figure(figsize=(10, 5))
        plt.style.use('ggplot')
        # Mock data for demonstration - in production, use your actual test results
        plt.scatter([1, 2, 3, 4, 5], [1.1, 2.1, 2.9, 4.2, 4.8], color='#2a5298', label='Model Trend')
        plt.plot([1, 5], [1, 5], color='red', linestyle='--', label='Ideal')
        plt.title('Actual vs Predicted Price Trend')
        plt.xlabel('Historical Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        
        # Convert plot to Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Metrics for display (Update these with real scores from your training notebook)
        metrics = {
            "r2_score": 0.89,
            "mae": "₹ 15,000",
            "mse": "0.021"
        }

        return render_template(
            'result.html',
            prediction=f"₹ {prediction:,.0f}",
            plot_url=plot_url,
            metrics=metrics
        )

if __name__ == '__main__':
    app.run(debug=True)