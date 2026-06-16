"""
=============================================================
  IRIS FLOWER CLASSIFICATION - Complete ML Pipeline
  Compatible with VS Code | Python 3.8+
  Dataset: Iris.csv (must be in the same folder as this script)
=============================================================

STEP-BY-STEP PROCEDURE
-----------------------
 Step 1 : Import Libraries
 Step 2 : Load & Explore the Dataset
 Step 3 : Preprocess the Data
 Step 4 : Split into Train / Test Sets
 Step 5 : Train Multiple ML Models
 Step 6 : Evaluate & Compare Models
 Step 7 : Visualize Results
 Step 8 : Predict on New / Custom Samples

REQUIRED LIBRARIES (install once):
  pip install pandas numpy scikit-learn matplotlib seaborn
"""

# ─────────────────────────────────────────────
# STEP 1 │ Import Libraries
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# Models we will compare
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier

print("=" * 60)
print("   IRIS FLOWER CLASSIFICATION - ML Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 2 │ Load & Explore the Dataset
# ─────────────────────────────────────────────
print("\n📂 STEP 2: Loading Dataset...")

# ── Load CSV (place Iris.csv in the same directory as this script) ──
df = pd.read_csv("Iris.csv")

print(f"\n✅ Dataset loaded successfully!")
print(f"   Shape : {df.shape[0]} rows × {df.shape[1]} columns")

print("\n── First 5 rows ──")
print(df.head())

print("\n── Dataset Info ──")
print(df.info())

print("\n── Basic Statistics ──")
print(df.describe())

print("\n── Class Distribution ──")
print(df["Species"].value_counts())

# Drop the 'Id' column – not a useful feature
df.drop(columns=["Id"], inplace=True)

# ─────────────────────────────────────────────
# STEP 3 │ Preprocess the Data
# ─────────────────────────────────────────────
print("\n🔧 STEP 3: Preprocessing...")

# Separate features (X) and target label (y)
X = df.drop(columns=["Species"])
y = df["Species"]

# Encode string labels → integers  (setosa=0, versicolor=1, virginica=2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_        # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Feature Scaling – standardise so all features have mean=0, std=1
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Features  : {list(X.columns)}")
print(f"   Classes   : {list(class_names)}")
print(f"   Encoded y : {np.unique(y_encoded)}  →  {list(class_names)}")

# ─────────────────────────────────────────────
# STEP 4 │ Split Dataset  (80 % train / 20 % test)
# ─────────────────────────────────────────────
print("\n✂️  STEP 4: Train / Test Split (80 / 20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded   # keeps class proportions equal in both splits
)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# STEP 5 │ Train Multiple Models
# ─────────────────────────────────────────────
print("\n🤖 STEP 5: Training Models...")

models = {
    "Logistic Regression" : LogisticRegression(max_iter=200),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"                 : SVC(kernel="rbf", probability=True),
    "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Accuracy on test set
    acc = accuracy_score(y_test, y_pred)
    # 5-fold cross-validation on full scaled dataset
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)

    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "accuracy" : acc,
        "cv_mean"  : cv_scores.mean(),
        "cv_std"   : cv_scores.std(),
    }
    print(f"   ✔  {name:<22}  Test Acc: {acc:.4f}  |  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# STEP 6 │ Evaluate Best Model in Detail
# ─────────────────────────────────────────────
print("\n📊 STEP 6: Detailed Evaluation...")

# Pick best model by test accuracy
best_name = max(results, key=lambda n: results[n]["accuracy"])
best      = results[best_name]
print(f"\n   🏆 Best Model : {best_name}  (Accuracy = {best['accuracy']:.4f})")

print(f"\n── Classification Report ({best_name}) ──")
print(classification_report(y_test, best["y_pred"], target_names=class_names))

# ─────────────────────────────────────────────
# STEP 7 │ Visualisations
# ─────────────────────────────────────────────
print("\n📈 STEP 7: Generating Visualisations...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Iris Flower Classification – Results Dashboard", fontsize=16, fontweight="bold")

# ── 7a  Pairplot (saved separately because seaborn needs its own figure) ──
pairplot_data = df.copy()
pairplot_data["Species"] = y          # original string labels
g = sns.pairplot(pairplot_data, hue="Species", diag_kind="kde", palette="Set2")
g.figure.suptitle("Pairplot of Iris Features by Species", y=1.02, fontsize=13)
plt.savefig("iris_pairplot.png", bbox_inches="tight", dpi=150)
plt.close()
print("   ✅ Saved: iris_pairplot.png")

# ── 7b  Model Accuracy Comparison bar chart ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Iris Classification – Analysis Dashboard", fontsize=15, fontweight="bold")

ax1 = axes[0, 0]
model_names = list(results.keys())
accuracies  = [results[n]["accuracy"] for n in model_names]
colors      = ["#4CAF50" if n == best_name else "#90CAF9" for n in model_names]
bars = ax1.barh(model_names, accuracies, color=colors, edgecolor="white")
ax1.set_xlim(0.85, 1.01)
ax1.set_xlabel("Test Accuracy")
ax1.set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accuracies):
    ax1.text(acc + 0.001, bar.get_y() + bar.get_height()/2,
             f"{acc:.4f}", va="center", fontsize=9)

# ── 7c  Confusion Matrix of best model ──
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, best["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title(f"Confusion Matrix – {best_name}")
ax2.tick_params(axis='x', rotation=20)

# ── 7d  Cross-Validation scores boxplot ──
ax3 = axes[1, 0]
cv_all = [cross_val_score(results[n]["model"], X_scaled, y_encoded, cv=5)
          for n in model_names]
bp = ax3.boxplot(cv_all, labels=[n.replace(" ", "\n") for n in model_names],
                 patch_artist=True,
                 boxprops=dict(facecolor="#90CAF9", color="#1565C0"),
                 medianprops=dict(color="#E53935", linewidth=2))
ax3.set_ylabel("CV Accuracy")
ax3.set_title("5-Fold Cross-Validation Distribution")
ax3.set_ylim(0.85, 1.02)

# ── 7e  Feature Importance (Random Forest) ──
ax4 = axes[1, 1]
rf_model      = results["Random Forest"]["model"]
importances   = rf_model.feature_importances_
feature_names = list(X.columns)
sorted_idx    = np.argsort(importances)[::-1]
ax4.bar([feature_names[i] for i in sorted_idx],
        [importances[i]  for i in sorted_idx],
        color="#FF8F00", edgecolor="white")
ax4.set_ylabel("Importance Score")
ax4.set_title("Feature Importances (Random Forest)")
ax4.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("iris_dashboard.png", bbox_inches="tight", dpi=150)
plt.show()
print("   ✅ Saved: iris_dashboard.png")

# ─────────────────────────────────────────────
# STEP 8 │ Predict on New Custom Samples
# ─────────────────────────────────────────────
print("\n🔮 STEP 8: Predicting on New Samples...")

# Each row = [SepalLength, SepalWidth, PetalLength, PetalWidth]
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],   # likely Iris-setosa
    [6.0, 2.9, 4.5, 1.5],   # likely Iris-versicolor
    [6.7, 3.0, 5.2, 2.3],   # likely Iris-virginica
])

# Scale using the SAME scaler fitted on training data
new_scaled = scaler.transform(new_samples)

# Use the best model to predict
best_model   = best["model"]
predictions  = best_model.predict(new_scaled)
pred_proba   = best_model.predict_proba(new_scaled)   # confidence scores

print(f"\n   Using model : {best_name}")
print(f"   {'Sample':<8} {'SepalL':>7} {'SepalW':>7} {'PetalL':>7} {'PetalW':>7}  │  {'Predicted Species':<22}  Confidence")
print("   " + "─" * 80)
for i, (sample, pred, proba) in enumerate(zip(new_samples, predictions, pred_proba)):
    species    = class_names[pred]
    confidence = proba.max() * 100
    print(f"   #{i+1:<7} {sample[0]:>7.1f} {sample[1]:>7.1f} {sample[2]:>7.1f} {sample[3]:>7.1f}  │  {species:<22}  {confidence:.1f}%")

# ─────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("   ✅  PIPELINE COMPLETE")
print("=" * 60)
print(f"   Best Model   : {best_name}")
print(f"   Test Accuracy: {best['accuracy'] * 100:.2f}%")
print(f"   CV Accuracy  : {best['cv_mean'] * 100:.2f}% ± {best['cv_std'] * 100:.2f}%")
print("\n   Output files saved in the same directory:")
print("   📄 iris_pairplot.png")
print("   📄 iris_dashboard.png")
print("=" * 60)