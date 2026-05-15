%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay)

df = pd.read_csv('bank-full.csv', sep=';')
print("Shape:", df.shape)

df = df.copy()
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('y', axis=1)
y = df['y']
print("Class distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc       = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall    = recall_score(y_test, y_pred) * 100
f1        = f1_score(y_test, y_pred) * 100
print(f"\nAccuracy:  {acc:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall:    {recall:.2f}%")
print(f"F1 Score:  {f1:.2f}%")

DARK='#1e293b'; BG='#0f172a'; TEXT='#e2e8f0'
MUTED='#94a3b8'; GRID='#334155'
RED='#ef4444'; GREEN='#22c55e'; BLUE='#38bdf8'; YELLOW='#facc15'

def style_ax(ax, title):
    ax.set_facecolor(DARK)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor(BG)

ax1 = axes[0, 0]
ax1.set_facecolor(DARK)
plot_tree(clf, max_depth=3, feature_names=X.columns.tolist(),
          class_names=['No','Yes'], filled=True, rounded=True,
          fontsize=7, ax=ax1)
ax1.set_title('Decision Tree Visualization (Depth 3)',
              color=TEXT, fontsize=11, fontweight='bold', pad=10)
for sp in ax1.spines.values(): sp.set_color(GRID)

ax2 = axes[0, 1]
importances = pd.Series(clf.feature_importances_, index=X.columns)
top10 = importances.nlargest(10).sort_values()
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10))
bars = ax2.barh(top10.index, top10.values, color=colors, edgecolor=GRID, zorder=3)
for bar, val in zip(bars, top10.values):
    ax2.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', color=TEXT, fontsize=8)
ax2.grid(axis='x', color=GRID, linestyle='--', alpha=0.4)
style_ax(ax2, 'Top 10 Feature Importances')

ax3 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No Purchase','Purchase'])
disp.plot(ax=ax3, colorbar=False, cmap='Greens')
ax3.set_facecolor(DARK)
ax3.set_title('Confusion Matrix', color=TEXT,
              fontsize=11, fontweight='bold', pad=10)
ax3.tick_params(colors=MUTED)
ax3.xaxis.label.set_color(MUTED)
ax3.yaxis.label.set_color(MUTED)
for sp in ax3.spines.values(): sp.set_color(GRID)

ax4 = axes[1, 1]
ax4.set_facecolor(DARK)
for sp in ax4.spines.values(): sp.set_color(GRID)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_vals  = [acc, precision, recall, f1]
bar_colors   = [GREEN, BLUE, YELLOW, '#a78bfa']
bars2 = ax4.bar(metric_names, metric_vals, color=bar_colors,
                edgecolor=GRID, width=0.5, zorder=3)
for bar, val in zip(bars2, metric_vals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}%', ha='center', color=TEXT,
             fontsize=11, fontweight='bold')
ax4.set_ylim(0, 110)
ax4.set_ylabel('Score (%)', color=MUTED)
ax4.grid(axis='y', color=GRID, linestyle='--', alpha=0.4)
ax4.tick_params(colors=MUTED)
ax4.set_title('Model Performance Metrics', color=TEXT,
              fontsize=11, fontweight='bold', pad=10)

fig.suptitle('Bank Marketing — Decision Tree Classifier',
             fontsize=20, fontweight='bold', color=TEXT, y=0.99)
fig.text(0.5, 0.965,
         'SkillCraft Technology | Data Science Internship | Task 3',
         ha='center', fontsize=10, color=MUTED, style='italic')

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('SCT_DS_3_decision_tree.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Done!")
