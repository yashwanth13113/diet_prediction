import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    df = pd.read_csv("healthy_diet_macros_by_height_weight.csv")
    df["Water_L"] = df["Weight_kg"] * 0.035
    X = df[["Height_cm", "Weight_kg"]]
    y = df[["Protein_g", "Carbs_g", "Fats_g", "Fiber_g", "Water_L"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    sample_input = pd.DataFrame([[13, 73]], columns=["Height_cm", "Weight_kg"])
    sample_prediction = model.predict(sample_input)[0]
    protein, carbs, fats, fiber, water = sample_prediction
    calories = (protein * 4) + (carbs * 4) + (fats * 9)
    print(f"Predicted Protein (g): {protein:.2f}")
    print(f"Predicted Carbs (g): {carbs:.2f}")
    print(f"Predicted Fats (g): {fats:.2f}")
    print(f"Predicted Fiber (g): {fiber:.2f}")
    print(f"Predicted Water (L): {water:.2f}")
    print(f"Estimated Calories (kcal): {calories:.2f}")

if __name__ == "__main__":
    main()
