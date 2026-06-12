# =====================================================================
# STEP 1: Import Libraries and Load Local CSV Data
# =====================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the data directly from your local file
# (This tells Python to read the file in your current folder)
df = pd.read_csv('Iris.csv')

print("--- Step 1: Previewing your local CSV data ---")
print(df.head(), "\n")

# =====================================================================
# STEP 2 & 3: Separate Features (X) and Target (y)
# =====================================================================
# X = The 4 physical measurements (Change these text strings if your CSV columns use different names!)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# y = The target column containing the flower species name or number
y = df['Species']

# =====================================================================
# STEP 4: Split Data into Training and Testing Sets
# =====================================================================
# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Model will study with: {len(X_train)} flowers.")
print(f"Model will take a test on: {len(X_test)} flowers.\n")

# =====================================================================
# STEP 5: Train the Model
# =====================================================================
# Initialize and train the Logistic Regression algorithm
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("--- Training complete! The model has learned from your CSV file. ---\n")

# =====================================================================
# STEP 6: Make Predictions and Evaluate
# =====================================================================
# Ask the model to guess the species of the test flowers
guesses = model.predict(X_test)

# Calculate the final accuracy score
final_accuracy = accuracy_score(y_test, guesses)

print("--- Final Results Using Local CSV ---")
print(f"Overall Accuracy: {final_accuracy * 100:.1f}%")
print("\nDetailed Report per Species:")
print(classification_report(y_test, guesses))