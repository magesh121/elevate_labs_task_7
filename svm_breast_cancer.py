import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("breast-cancer.csv")
print("\n📊 Dataset Shape:", df.shape)
print("\n🔍 Dataset Preview:")
print(df.head())

# Encode categorical labels if any
if df['diagnosis'].dtype == 'object':
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Features and target
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
y = df['diagnosis']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 1️⃣ SVM with Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

# 2️⃣ SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

# 3️⃣ Hyperparameter tuning with GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
best_svm = grid.best_estimator_
y_pred_best = best_svm.predict(X_test)

# 🎯 Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n📌 Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"outputs/conf_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()

# ✅ Evaluate All Models
evaluate_model(y_test, y_pred_linear, "SVM Linear")
evaluate_model(y_test, y_pred_rbf, "SVM RBF")
evaluate_model(y_test, y_pred_best, "Tuned SVM RBF")

# 📊 Accuracy Comparison
accuracies = [accuracy_score(y_test, y_pred_linear),
              accuracy_score(y_test, y_pred_rbf),
              accuracy_score(y_test, y_pred_best)]

models = ["SVM Linear", "SVM RBF", "Tuned SVM RBF"]
plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.001, f"{v:.3f}", ha='center', fontweight='bold')
plt.savefig("outputs/model_accuracy_comparison.png")
plt.close()

# 🔍 PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
svm_vis = SVC(kernel='linear')
svm_vis.fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Decision Boundary with PCA (Linear SVM)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("outputs/svm_pca_decision_boundary.png")
plt.close()

# 📈 Additional Graphs
# 1. Class Distribution
plt.figure()
df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, labels=['Benign', 'Malignant'])
plt.title("Class Distribution")
plt.ylabel('')
plt.savefig("outputs/class_distribution_pie.png")
plt.close()

# 2. Feature Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pd.DataFrame(X, columns=X.columns).corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/feature_correlation_heatmap.png")
plt.close()

# 3. Pairplot of Top Features
top_features = df.corr()['diagnosis'].abs().sort_values(ascending=False).index[1:4].tolist()
sns.pairplot(df[top_features + ['diagnosis']], hue='diagnosis', palette='husl')
plt.suptitle("Pairplot of Top Correlated Features", y=1.02)
plt.savefig("outputs/pairplot_top_features.png")
plt.close()

# 4. Boxplot of Radius Mean by Diagnosis
plt.figure(figsize=(8, 5))
sns.boxplot(x='diagnosis', y='radius_mean', data=df, palette='Set2')
plt.title("Radius Mean by Diagnosis")
plt.savefig("outputs/boxplot_radius_mean.png")
plt.close()

# 5. Violin Plot of Texture Mean
plt.figure(figsize=(8, 5))
sns.violinplot(x='diagnosis', y='texture_mean', data=df, palette='muted')
plt.title("Texture Mean by Diagnosis")
plt.savefig("outputs/violinplot_texture_mean.png")
plt.close()

# ✅ Summary
print("\n🎉 Task 7 complete. All plots saved in the 'outputs' folder.")
