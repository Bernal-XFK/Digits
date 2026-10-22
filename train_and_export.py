import os
import json
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import ndimage
from PIL import Image, ImageOps

# Ensure output directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

print("--- [1/6] Cargando y explorando el dataset UCI Digits ---")
digits = load_digits()
X_raw = digits.data  # (1797, 64) en rango 0..16
y = digits.target    # (1797,)
images = digits.images # (1797, 8, 8)

print(f"Total de muestras: {X_raw.shape[0]}")
print(f"Dimensiones de cada imagen: 8x8 ({X_raw.shape[1]} píxeles)")
print(f"Clases únicas: {np.unique(y)}")
print(f"Rango de intensidad de píxeles: [{X_raw.min()}, {X_raw.max()}]")

# 1. Visualización de ejemplos por cada clase (0-9)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit_val in range(10):
    ax = axes[digit_val // 5, digit_val % 5]
    idx = np.where(y == digit_val)[0][0]
    im = ax.imshow(images[idx], cmap='Blues', interpolation='nearest')
    ax.set_title(f"Dígito: {digit_val}", fontsize=12, fontweight='bold', color='#1e293b')
    ax.axis('off')
plt.suptitle("Muestras Representativas del Dataset UCI Digits (8x8 píxeles)", fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
plt.tight_layout()
sample_grid_path = os.path.join(ASSETS_DIR, "sample_digits_grid.png")
plt.savefig(sample_grid_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico guardado: {sample_grid_path}")

# 2. Preprocesamiento: Normalización
X_norm = X_raw / 16.0  # Rango [0.0, 1.0]

# Partición estratificada Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=2025, stratify=y
)
print(f"Partición de entrenamiento: {X_train.shape[0]} muestras")
print(f"Partición de prueba: {X_test.shape[0]} muestras")

# 3. Análisis de Componentes Principales (PCA)
print("\n--- [2/6] Análisis de Componentes Principales (PCA) ---")
pca_full = PCA().fit(X_train)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Scree plot
ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker='o', color='#2563eb', linewidth=2, markersize=4)
ax1.axhline(90, color='#ea580c', linestyle='--', label=f'90% Varianza ({n_components_90} componentes)')
ax1.axhline(95, color='#16a34a', linestyle='--', label=f'95% Varianza ({n_components_95} componentes)')
ax1.set_xlabel('Número de Componentes Principales', fontsize=12)
ax1.set_ylabel('Varianza Explicada Acumulada (%)', fontsize=12)
ax1.set_title('Curva de Varianza Acumulada (Scree Plot)', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', frameon=True)
ax1.grid(True, alpha=0.3)

# Proyección 2D
pca_2d = PCA(n_components=2, random_state=2025)
X_train_pca2 = pca_2d.fit_transform(X_train)
scatter = ax2.scatter(X_train_pca2[:, 0], X_train_pca2[:, 1], c=y_train, cmap='tab10', alpha=0.7, s=20)
cbar = plt.colorbar(scatter, ax=ax2, ticks=range(10))
cbar.set_label('Dígito', fontsize=11)
ax2.set_xlabel(f'Componente 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax2.set_ylabel(f'Componente 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax2.set_title('Proyección 2D de los Dígitos con PCA', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
pca_chart_path = os.path.join(ASSETS_DIR, "pca_analysis.png")
plt.savefig(pca_chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico guardado: {pca_chart_path}")

# 4. Definición y Entrenamiento de Modelos
print("\n--- [3/6] Entrenamiento y Comparación de Algoritmos ---")
models = {
    "SVM (RBF)": SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, random_state=2025),
    "MLP (Red Neuronal)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, activation='relu', solver='adam', alpha=0.001, random_state=2025),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15, random_state=2025),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=2025)
}

results = []
trained_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

for name, clf in models.items():
    print(f"Entrenando {name}...")
    # Pipeline sin PCA y con PCA para medir impacto
    # Note: SVM on full 64 features works exceptionally well (>98% acc)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Inferencia en Test
    t_inf0 = time.time()
    y_pred = clf.predict(X_test)
    inf_time_per_sample = ((time.time() - t_inf0) / len(X_test)) * 1000  # ms
    
    acc_test = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    
    trained_models[name] = clf
    results.append({
        "Model": name,
        "Test Accuracy": acc_test,
        "CV Mean Accuracy": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Training Time (s)": train_time,
        "Inference Time (ms/sample)": inf_time_per_sample,
        "y_pred": y_pred
    })
    print(f" -> Accuracy Test: {acc_test*100:.2f}% | CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}% | Tiempo: {train_time:.3f}s")

# 5. Gráfico de Comparación de Modelos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
names = [r["Model"] for r in results]
accs = [r["Test Accuracy"] * 100 for r in results]
times = [r["Training Time (s)"] for r in results]

colors = ['#10b981' if a == max(accs) else '#3b82f6' for a in accs]
bars1 = ax1.barh(names, accs, color=colors, edgecolor='#1e293b', height=0.6)
ax1.set_xlim(80, 101)
ax1.set_xlabel('Exactitud en Test (%)', fontsize=12, fontweight='bold')
ax1.set_title('Comparativa de Exactitud (Test Accuracy)', fontsize=13, fontweight='bold')
for bar in bars1:
    width = bar.get_width()
    ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", 
             va='center', fontsize=10, fontweight='bold', color='#1e293b')

bars2 = ax2.barh(names, times, color='#f59e0b', edgecolor='#1e293b', height=0.6)
ax2.set_xlabel('Tiempo de Entrenamiento (segundos)', fontsize=12, fontweight='bold')
ax2.set_title('Eficiencia Computacional (Tiempo de Entrenamiento)', fontsize=13, fontweight='bold')
for bar in bars2:
    width = bar.get_width()
    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.3f}s", 
             va='center', fontsize=10, fontweight='bold', color='#1e293b')

plt.tight_layout()
benchmark_chart_path = os.path.join(ASSETS_DIR, "model_benchmark.png")
plt.savefig(benchmark_chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico guardado: {benchmark_chart_path}")

# 6. Matrices de Confusión de los 4 mejores modelos
top_4 = sorted(results, key=lambda x: x["Test Accuracy"], reverse=True)[:4]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, res in enumerate(top_4):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title(f"{res['Model']} (Acc: {res['Test Accuracy']*100:.2f}%)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Clase Predicha', fontsize=11)
    ax.set_ylabel('Clase Real', fontsize=11)

plt.suptitle("Matrices de Confusión en Partición de Test (360 muestras)", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
cm_chart_path = os.path.join(ASSETS_DIR, "confusion_matrices.png")
plt.savefig(cm_chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico guardado: {cm_chart_path}")

# 7. Guardar Modelos Serializados y Reporte
best_result = top_4[0]
best_model_name = best_result["Model"]
best_model = trained_models[best_model_name]
print(f"\nModelo Campeón: {best_model_name} con {best_result['Test Accuracy']*100:.2f}% de exactitud.")

# Serializar el modelo campeón y el MLP para la interfaz
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_svm_model.joblib"))
joblib.dump(trained_models["MLP (Red Neuronal)"], os.path.join(MODELS_DIR, "best_mlp_model.joblib"))
joblib.dump(trained_models["Random Forest"], os.path.join(MODELS_DIR, "best_rf_model.joblib"))
print("Modelos serializados con éxito en carpeta 'models/'.")

# Guardar resumen en JSON
benchmark_summary = {
    "author": "José Alejandro Bernal Figueroa",
    "project": "Digits - Reconocimiento Óptico de Dígitos Manuscritos",
    "career": "Ingeniería de Sistemas",
    "dataset": "UCI Optical Recognition of Handwritten Digits (8x8)",
    "total_samples": int(X_raw.shape[0]),
    "features": int(X_raw.shape[1]),
    "test_samples": int(X_test.shape[0]),
    "best_model": best_model_name,
    "best_accuracy": float(best_result["Test Accuracy"]),
    "models_evaluated": [
        {
            "name": r["Model"],
            "test_accuracy": float(r["Test Accuracy"]),
            "cv_accuracy_mean": float(r["CV Mean Accuracy"]),
            "cv_accuracy_std": float(r["CV Std"]),
            "train_time_sec": float(r["Training Time (s)"]),
            "inference_time_ms": float(r["Inference Time (ms/sample)"])
        }
        for r in results
    ]
}

with open(os.path.join(ASSETS_DIR, "benchmark_summary.json"), "w", encoding="utf-8") as f:
    json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)

print("\n--- Pipeline de Entrenamiento y Exportación finalizado con éxito ---")
