import cv2
import numpy as np
from flask import Flask, render_template, request
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained models for acetaldehyde and acetone
acetaldehyde_models = {
    "Bromocresol Purple": joblib.load("models/acetaldehyde_bromocresol_purple_xgboost_model.pkl"),
    "Fluorescein": joblib.load("models/acetaldehyde_fluorescein_xgboost_model.pkl"),
    "Bromophenol Blue": joblib.load("models/acetaldehyde_bromophenol_blue_xgboost_model.pkl"),
    "Crystal Violet": joblib.load("models/acetaldehyde_crystal_violet_xgboost_model.pkl"),
    "Bromocresol Green": joblib.load("models/acetaldehyde_bromocresol_green_xgboost_model.pkl")
}

acetone_models = {
    "Bromocresol Green": joblib.load("models/acetone_bromocresol_green_xgboost_model.pkl"),
    "Bromophenol Blue": joblib.load("models/acetone_bromophenol_blue_xgboost_model.pkl"),
    "Pyrocatechol Violet": joblib.load("models/acetone_pyrocatechol_violet_xgboost_model.pkl"),
    "Chlorophenol Red": joblib.load("models/acetone_chlorophenol_red_xgboost_model.pkl"),
    "Bromocresol Purple": joblib.load("models/acetone_bromocresol_purple_xgboost_model.pkl")
}

# Initial RGB values for each dye (for manual input)
initial_rgb_values_manual = {
    "Bromocresol Purple": (101.317, 122.143, 27.849),
    "Fluorescein": (200.216, 210.148, 17.321),
    "Bromophenol Blue": (30.328, 33.692, 87.031),
    "Crystal Violet": (38.834, 17.696, 121.416),
    "Bromocresol Green": (14.814, 56.779, 63.313),
    "Pyrocatechol Violet": (170.812, 122.533, 5.645),
    "Chlorophenol Red": (45.717, 35.687, 28.429),
}

# Initial RGB values for each dye (for image input)
initial_rgb_values_image = {
    "acetaldehyde": {
        "Bromocresol Purple": (108.39, 132.26, 33.79),
        "Fluorescein": (205.97, 214.92, 25.85),
        "Bromophenol Blue": (30.92,38.90, 100.77),
        "Crystal Violet": (40.79, 16.90, 134.34),
        "Bromocresol Green": (16.97, 67.91, 78.90)
    },
    "acetone": {
        "Bromocresol Green": (16.97, 67.91, 78.90),
        "Bromophenol Blue": (30.92,38.90, 100.77),
        "Pyrocatechol Violet": (156.78,117.83, 7.98),
        "Chlorophenol Red": (47.70,43.72, 34.77),
        "Bromocresol Purple": (108.39, 132.26, 33.79)
    }
}

# Sensitivity formula
def calculate_sensitivity(delta_r, delta_g, delta_b, initial_r, initial_g, initial_b):
    return (delta_r + delta_g + delta_b) * 100 / (initial_r + initial_g + initial_b)

dye_names = [
    "Methyl Red", "Bromocresol Purple", "Chlorophenol Red", "Bromocresol Green",
    "Bromophenol Blue", "Methyl Orange", "Metanil Yellow", "Crystal Violet",
    "Indigo Carmine", "Toluidine Blue", "Fluorescein", "Methyl Violet",
    "Basic Fuchsin", "Cresol Red", "Bromothymol Blue", "Thymol Blue", "Neutral Red",
    "Pyrocatechol Violet", "m-cresol purple", "Congo Red", "Alizarin",
    "Acridine Orange", "Methylthymol Blue", "Nitrazine Yellow", "Phenol Red"
]

# Function to extract RGB values from an image and store them with dye names
def extract_rgb_from_image(image_path, selected_dots):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=24
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        sorted_circles = sorted(circles[0, :], key=lambda c: c[1])
        row_tolerance = 20
        rows = []
        current_row = [sorted_circles[0]]
        for circle in sorted_circles[1:]:
            if abs(circle[1] - current_row[-1][1]) <= row_tolerance:
                current_row.append(circle)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [circle]
        rows.append(sorted(current_row, key=lambda c: c[0]))

        sorted_circles = [circle for row in rows for circle in row]

        rgb_values = {}
        for i, circle in enumerate(sorted_circles):
            x, y, r = circle
            mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_color = cv2.mean(image_rgb, mask=mask)[:3]
            rgb_values[dye_names[i]] = mean_color  # Store RGB values with dye names
            print(f"Extracted RGB for {dye_names[i]}: {mean_color}")

        # Select the relevant dots for the model
        selected_rgb = {}
        for i in selected_dots:
            dye_name = dye_names[i]
            if dye_name in rgb_values:
                selected_rgb[dye_name] = rgb_values[dye_name]
            else:
                print(f"Warning: {dye_name} not found in rgb_values")

        return selected_rgb
    else:
        return {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select', methods=['GET'])
def select_option():
    option = request.args.get('option')  # 'acetaldehyde' or 'acetone'
    return render_template('select_dye.html', option=option)

@app.route('/predict', methods=['POST'])
def predict():
    option = request.form.get('option')
    dye = request.form.get('dye')
    delta_r = request.form.get('delta_r')
    delta_g = request.form.get('delta_g')
    delta_b = request.form.get('delta_b')

    if delta_r and delta_g and delta_b:  # If RGB values are manually entered
        delta_r = float(delta_r)
        delta_g = float(delta_g)
        delta_b = float(delta_b)

        # Get initial RGB values for the selected dye (for manual data)
        initial_r, initial_g, initial_b = initial_rgb_values_manual[dye]

        # Calculate sensitivity
        sensitivity = calculate_sensitivity(delta_r, delta_g, delta_b, initial_r, initial_g, initial_b)

        # Predict concentration using the appropriate model
        models = acetaldehyde_models if option == 'acetaldehyde' else acetone_models
        model = models[dye]
        new_data = [[delta_r, delta_g, delta_b, sensitivity]]  # Use the features
        concentration = model.predict(new_data)[0]

        # Ensure concentration is not negative
        concentration = max(0, concentration)

        # Format concentration to 3 decimal places
        concentration = round(concentration, 3)

        # Determine spoilage status
        if option == 'acetaldehyde':
            if concentration > 25:
                status = "Spoiled"
            elif 15 < concentration <= 25:
                status = "Intermediate"
            else:
                status = "Fresh"
        else:  # For acetone
            if concentration > 6:
                status = "Spoiled"
            elif 3 < concentration <= 6:
                status = "Intermediate"
            else:
                status = "Fresh"

        return render_template('result.html', option=option, dye=dye, concentration=concentration, status=status)


    else:  # If the image upload option is selected

        image = request.files.get('image')

        if image:

            image_path = 'static/temp_image.jpg'

            image.save(image_path)

            # Extract RGB values from the uploaded image based on the selected dye

            selected_dots = [dye_names.index(dye)]  # Only the selected dye's dot

            # Extract RGB values from the uploaded image

            extracted_rgb = extract_rgb_from_image(image_path, selected_dots)

            if not extracted_rgb:
                return "No circles detected in the image."

            # Get initial RGB values for the selected dye (for image data)

            initial_rgb = initial_rgb_values_image[option][dye]

            # Calculate delta values and sensitivity for the selected dye

            delta_r, delta_g, delta_b = 0, 0, 0

            if dye in extracted_rgb:
                extracted = extracted_rgb[dye]

                delta_r = abs(initial_rgb[0] - extracted[0])

                delta_g = abs(initial_rgb[1] - extracted[1])

                delta_b = abs(initial_rgb[2] - extracted[2])

            # Calculate sensitivity

            sensitivity = calculate_sensitivity(delta_r, delta_g, delta_b, *initial_rgb)

            # Predict concentration using the appropriate model

            models = acetaldehyde_models if option == 'acetaldehyde' else acetone_models

            model = models[dye]

            new_data = [[delta_r, delta_g, delta_b, sensitivity]]  # Use the features

            # Predict concentration using the appropriate model
            concentration = model.predict(new_data)[0]

            # Ensure concentration is not negative
            concentration = max(0, concentration)

            # Format concentration to 3 decimal places
            concentration = round(concentration, 3)

            # Determine spoilage status
            if option == 'acetaldehyde':
                if concentration > 25:
                    status = "Spoiled"
                elif 15 < concentration <= 25:
                    status = "Intermediate"
                else:
                    status = "Fresh"
            else:  # For acetone
                if concentration > 6:
                    status = "Spoiled"
                elif 3 < concentration <= 6:
                    status = "Intermediate"
                else:
                    status = "Fresh"

            return render_template('result.html', option=option, dye=dye, concentration=concentration, status=status)


if __name__ == '__main__':
    app.run(debug=True)
