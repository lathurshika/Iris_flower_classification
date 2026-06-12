# ==========================================
# STEP 1 & 2: Import tools and load the data
# ==========================================
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the built-in dataset
iris_data = load_iris()

# Let's put it in a clean table (DataFrame) just to preview it
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target
print("--- A quick look at the first 5 rows of data ---")
print(df.head(), "\n")

# ==========================================
# STEP 3 & 4: Separate and Split the data
# ==========================================
X = iris_data.data    # The 4 flower measurements
y = iris_data.target  # The 3 flower species numbers

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Model will study with: {len(X_train)} flowers.")
print(f"Model will take a test on: {len(X_test)} flowers.\n")

# ==========================================
# STEP 5: Choose a model and Train it
# ==========================================
# We initialize the algorithm (max_iter gives it enough time to find patterns)
flower_classifier = LogisticRegression(max_iter=200)

# .fit() is the command that triggers the actual learning/training
flower_classifier.fit(X_train, y_train)
print("--- Training complete! The model has learned the patterns. ---\n")

# ==========================================
# STEP 6: Test the model and see how it did
# ==========================================
# Ask the model to guess the species of the test flowers
guesses = flower_classifier.predict(X_test)

# Calculate the final grade
final_accuracy = accuracy_score(y_test, guesses)

print("--- Final Exam Results ---")
print(f"Overall Accuracy: {final_accuracy * 100:.1f}%")
print("\nDetailed Report per Species:")
print(classification_report(y_test, guesses, target_names=iris_data.target_names))