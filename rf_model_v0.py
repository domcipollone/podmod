from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import xgboost as xgb 
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from joblib import dump 
from datetime import datetime


print('reading_csv...')
df = pd.read_csv("feature_analysis/features_20250725.csv", header=0)
df['is_ad'] = (df['target'] == 'ad').astype(int)

correlated_features = ['spectral_rolloff_85_mean', 'chroma_var', 'amplitude_mean', 'amplitude_std', 'spectral_rolloff_95_mean', 'dynamic_range', 'spectral_rolloff_mean']
correlated_features_v2 = ['poly_features_mean', 'zcr_std', 'beats_per_second', 'poly_features_std', 'amplitude_min', 'mfcc_2_mean', 'spectral_flatness_std', 'rms_std']

non_features = ['audio_file','transcript_id','start_time','end_time','duration','confidence','target']
features_to_drop = correlated_features + correlated_features_v2

df.drop(non_features, axis=1, inplace=True)
df.drop(features_to_drop, axis=1, inplace=True)

df['tempo'] = df['tempo'].str.extract(r'(\d+\.?\d*)').astype(float)

df_targets = df[df['is_ad'] == 1].copy()
df_non_targets = df[df['is_ad'] == 0].sample(n=len(df_targets)*4).copy()

df_model = pd.concat([df_targets, df_non_targets], axis=0)

print('splitting into X and y')
y = df_model['is_ad']

X = df_model.drop(['is_ad'], axis=1, inplace=False)

print(f"{len(X)} samples in loaded dataset")
print(f"{sum(y)} ads; {len(y) - sum(y)} content segments")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

print('fitting classifier...')

pipeline = Pipeline(steps=[('xgb', xgb.XGBClassifier())])

xgb_params = {'xgb__eta': np.linspace(0.05, 0.5, 10), 
              'xgb__max_depth': np.arange(4, 9, 1),
              'xgb__reg_lambda': np.arange(0, 5, 1), 
              'xgb__reg_alpha': np.arange(0, 3, 1), 
              'xgb__scale_pos_weight': np.arange(1, 5, 1)
              }

xgb_cv = RandomizedSearchCV(estimator=pipeline,
                            param_distributions=xgb_params, 
                            scoring='precision', 
                            refit=True,
                            n_jobs=6, 
                            random_state=22, 
                            return_train_score=True, 
                            verbose=2, 
                            n_iter=20, 
                            cv=5
                            )

xgb_cv.fit(X_train, y_train)

best_model = xgb_cv.best_estimator_

print('plotting confusion matrix')

predictions = best_model.predict(X_test)
scores = best_model.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test, predictions)
precision, recall, _ = precision_recall_curve(y_test, scores)

fpr, tpr, _ = roc_curve(y_test, scores)

disp_pr = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()

plt.show()

dump(best_model, 'feature_analysis/models/xgb_best_model_20250727.joblib')
