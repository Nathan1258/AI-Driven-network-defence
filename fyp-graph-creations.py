
# Code to find feature importance

from sklearn.ensemble import RandomForestClassifier
import numpy as np

rf_model = RandomForestClassifier(n_estimators=25, verbose=2, n_jobs=-1) 
rf_model.fit(X_train, Y_train)  


importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)


aggregated_importances = {}
for i, feature in enumerate(X_train.columns):
    if feature.startswith('Protocol_') or feature.startswith('Connection_State_'):
        root_feature = feature.split('_')[0]
        aggregated_importances.setdefault(root_feature, 0)
        aggregated_importances[root_feature] += importances[i]
    else:
        aggregated_importances[feature] = importances[i]

print("Feature Ranking:")
for feature, importance in sorted(aggregated_importances.items(), key=lambda item: item[1], reverse=True):
    print("%d. Feature %s (%f)" % (len(aggregated_importances) - aggregated_importances.keys().index(feature), feature, importance))


#  Code to create plot of feature importance
import matplotlib.pyplot as plt

features = sorted(aggregated_importances, key=aggregated_importances.get, reverse=True)
importances = [aggregated_importances[feature] for feature in features]
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Ranking - Random Forest")
plt.tight_layout()
plt.show()


# Grouped Bar chart code for all metrics
import numpy as np
import matplotlib.pyplot as plt

methods = ['Baseline', 'Resampled', 'Reduced Features 1', 'Reduced Features 2', 'PCA (2 components)',
           'PCA (3 components)', 'ICA (3 components)', 'LDA']
precision = np.array([
    [91.77, 99.99, 70.22, 75.83],
    [95.60, 99.99, 92.15, 97.97],
    [95.06, 96.05, 91.94, 97.94],
    [95.54, 99.99, 91.87, 97.94],
    [25.00, 0, 88.20, 0],
    [50.48, 49.36, 83.33, 0],
    [46.54, 81.39, 0, 87.05],
    [99.23, 96.18, 92.12, 97.45]
])
recall = np.array([
    [91.22, 99.99, 71.14, 75.20],
    [92.06, 99.99, 94.01, 99.61],
    [87.75, 98.83, 93.82, 96.61],
    [91.76, 99.99, 93.93, 99.62],
    [99.99, 0, 0, 0],
    [96.67, 99.99, 8.78, 0],
    [85.07, 95.98, 0, 82.15],
    [88.51, 99.93, 97.21, 96.03]
])
f1_score = np.array([
    [91.49, 99.99, 70.81, 75.51],
    [93.80, 99.99, 93.07, 98.78],
    [92.26, 97.90, 92.87, 98.77],
    [93.61, 99.99, 92.89, 98.77],
    [40.00, 0, 0, 0],
    [67.02, 66.09, 1.75, 0],
    [60.17, 89.73, 0, 84.53],
    [89.66, 98.05, 90.03, 96.73]
])

bar_width = 0.2

r1 = np.arange(len(methods))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

colors = ['#e60049', '#9b19f5', '#ffa300']

plt.bar(r1, precision[:, 0], color=colors[0], width=bar_width, edgecolor='grey', label='Precision')
plt.bar(r2, recall[:, 0], color=colors[1], width=bar_width, edgecolor='grey', label='Recall')
plt.bar(r3, f1_score[:, 0], color=colors[2], width=bar_width, edgecolor='grey', label='F1-Score')

plt.xlabel('Methods', fontweight='bold')
plt.xticks([r + bar_width * 1.5 for r in range(len(methods))], methods, rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.show()

# Confusion Matrix code to use custom labels and add percentages instead of values for easier readability.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

conf_matrix = np.array([[1340177, 2, 361042, 5676],
                        [3, 1706788, 77, 28],
                        [10346, 50, 1659318, 37183],
                        [0, 67631, 5862, 1639265]])

labels = ["Benign", "Port Scan", "DDoS", "Okiru"]

conf_matrix_percentages = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_percentages, annot=True, fmt='.2%', linewidths=0.5, linecolor="red", ax=ax,
            xticklabels=labels, yticklabels=labels)

plt.show()