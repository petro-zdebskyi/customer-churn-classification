import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score


def get_features_and_target(data):
    plans = ['Basic', 'Standard', 'Premium']
    cus_ids = list(set(data['customer_id']))
    features_list = []
    is_churned_list = []
    for cus_id in cus_ids:
        d = data[data['customer_id'] == cus_id].sort_values(by='date')

        # handling first timestamp for the user
        curr_plan_onehot = [0]*3
        curr_plan_onehot[plans.index(d.iloc[0]['plan_type'])] = 1
        curr_transaction = d.iloc[0]['transaction_amount']
        curr_month = int(d.iloc[0]['date'].split('-')[1])
        curr_day = int(d.iloc[0]['date'].split('-')[2])
        curr_cpi = df_cpi.loc[df_cpi['date'] == d.iloc[0]['date'], 'cpi'].iloc[0]
        features_list.append([*curr_plan_onehot, curr_transaction, curr_month, curr_day, curr_cpi, 0, 0, 0, 0])
        is_churned_list.append(d.iloc[0]['churn'])

        # handling the rest
        for idx in range(1, len(d)):
            curr_plan_onehot = [0]*3
            curr_plan_onehot[plans.index(d.iloc[idx]['plan_type'])] = 1
            curr_transaction = d.iloc[idx]['transaction_amount']
            curr_month = int(d.iloc[idx]['date'].split('-')[1])
            curr_day = int(d.iloc[idx]['date'].split('-')[2])
            curr_cpi = df_cpi.loc[df_cpi['date'] == d.iloc[idx]['date'], 'cpi'].iloc[0]
            prev_transactions_mean = d.iloc[:idx]['transaction_amount'].mean()
            prev_transactions_std = d.iloc[:idx]['transaction_amount'].std() if idx > 1 else 0
            prev_plan_types_count = len(set(d.iloc[:idx]['plan_type']))
            prev_churned_count = np.count_nonzero(d.iloc[:idx]['churn'] == 1)
            features_list.append([
                *curr_plan_onehot,
                curr_transaction,
                curr_month,
                curr_day,
                curr_cpi,
                prev_transactions_mean,
                prev_transactions_std,
                prev_plan_types_count,
                prev_churned_count
            ])
            is_churned_list.append(d.iloc[idx]['churn'])
    feat_names = np.array([
        'curr_plan_onehot',
        'curr_plan_onehot',
        'curr_plan_onehot',
        'curr_transaction',
        'curr_month',
        'curr_day',
        'curr_cpi',
        'prev_transactions_mean',
        'prev_transactions_std',
        'prev_plan_types_count',
        'prev_churned_count'
    ])
    return np.array(features_list), np.array(is_churned_list), feat_names


# Read churn dataframe
df = pd.read_csv('data/input/churn_data.csv')

# Consumer Price Index - external source to enrich data
df_cpi = pd.read_csv('data/input/CPIAUCSL.csv')
df_cpi.columns = ['date', 'cpi']  # renaming columns for clarity

# Train test split
customer_ids = list(set(df['customer_id']))
train_set_ratio = 0.6  # 60% for test, the rest 50/50 for validation and test

train_customer_ids = np.random.choice(customer_ids, int(len(customer_ids)*train_set_ratio), replace=False)

non_train_customer_ids = [i for i in customer_ids if i not in train_customer_ids]
val_customer_ids = np.random.choice(
    non_train_customer_ids,
    int(len(non_train_customer_ids)*.5),
    replace=False
)

test_customer_ids = [i for i in customer_ids if i not in train_customer_ids and i not in val_customer_ids]

df_train = df[df['customer_id'].isin(train_customer_ids)]
df_val = df[df['customer_id'].isin(val_customer_ids)]
df_test = df[df['customer_id'].isin(test_customer_ids)]

# Handling missing values
train_transaction_median = df_train['transaction_amount'].median()
train_popular_plan = df_train['plan_type'].mode()[0]

for df_chunk in [df_train, df_val, df_test]:
    mask = df_chunk['transaction_amount'].isna()
    if np.count_nonzero(mask):
        df_chunk.loc[mask, 'transaction_amount'] = train_transaction_median

    mask = df_chunk['plan_type'].isna()
    if np.count_nonzero(mask):
        df_chunk.loc[mask, 'plan_type'] = train_popular_plan

# Generate features
train_features, train_target, feature_names = get_features_and_target(df_train)
val_features, val_target, _ = get_features_and_target(df_val)
test_features, test_target, _ = get_features_and_target(df_test)

# Scale features (not required for RandomForest)
scaler = StandardScaler()
scaler.fit(train_features)
train_features = scaler.transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Balancing data
resulting_groups_size = np.count_nonzero(train_target == 1)*2

train_features_churn = np.concatenate((train_features[train_target == 1], train_features[train_target == 1]), axis=0)
train_target_churn = [1]*resulting_groups_size

rnd_indices = np.random.choice(len(train_features[train_target == 0]), resulting_groups_size, replace=False)
train_features_non_churn = train_features[train_target == 0][rnd_indices]
train_target_non_churn = [0]*resulting_groups_size

new_indices = np.arange(resulting_groups_size*2)
np.random.shuffle(new_indices)

train_features = np.concatenate((train_features_churn, train_features_non_churn), axis=0)
train_target = np.concatenate((train_target_churn, train_target_non_churn), axis=0)
train_features = train_features[new_indices]
train_target = train_target[new_indices]

X_train, y_train = train_features, train_target
X_val, y_val = val_features, val_target
X_test, y_test = test_features, test_target

# Perform random search for parameters
n_iter = 10  # number of random samples to try
random_n_estimators = np.random.randint(50, 300, size=n_iter)
random_max_depth = np.random.choice([None, 10, 20], size=n_iter)
max_features = np.random.choice([None, 'sqrt', 'log2', 0.3, 0.5, 0.7], size=n_iter)
best_model = None
best_score = -np.inf
best_params = None

for i in range(n_iter):
    params = {
        'n_estimators': random_n_estimators[i],
        'max_depth': random_max_depth[i],
        'max_features': max_features[i]
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_search = model.predict(X_val)
    score = f1_score(y_val, y_pred_search)
    if score > best_score:
        best_model = model
        best_score = score
        best_params = params

print(f'Best model: {best_model}')
print(f'Best params: {best_params}')
print(f'Best F1 on validation: {best_score:.4f}')
feat_imp_idxs = np.argsort(best_model.feature_importances_)[::-1]
print(f'Feature importance: {list(zip(feature_names[feat_imp_idxs], best_model.feature_importances_[feat_imp_idxs]))}')

# Evaluating best model on test set
clf = best_model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 score: {2*precision*recall / (precision+recall):.2f}')

sample = X_test[0:1]
prediction = clf.predict(sample)
print(f'Sample prediction: {prediction}')

# Saving evaluation and model into files
with open('data/output/evaluation.json', 'w') as f:
    json.dump({'precision': precision, 'recall': recall}, f)

with open('data/output/classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Saving dataframe with prediction column
df_in = df.copy()
df_in.loc[df_in['transaction_amount'].isna(), 'transaction_amount'] = train_transaction_median
df_in.loc[df_in['plan_type'].isna(), 'plan_type'] = train_popular_plan

out_features, _, _ = get_features_and_target(df_in)
out_features = scaler.transform(out_features)
out_pred = clf.predict(out_features)
df_out = df.copy()
df_out['churn_prediction'] = out_pred

df_out.to_csv('data/output/churn_data_predicted.csv', index=False)
