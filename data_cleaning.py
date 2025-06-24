import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected line
df = pd.read_csv('mental_health.csv')

# Normalize gender
df['Gender'] = df['Gender'].str.strip().str.lower()

# Replace known variants
df['Gender'] = df['Gender'].replace({
    'm': 'male', 'f': 'female',
    'trans-female': 'trans-female',
    'trans male': 'trans-male',
    'non-binary': 'non-binary',
    'queer': 'queer',
    'male': 'male',  # include lowercase too
    'female': 'female',
    'femake': 'female'
})

rare_genders = df['Gender'].value_counts()[df['Gender'].value_counts() < 5].index
df['Gender'] = df['Gender'].apply(lambda x: 'other' if x in rare_genders else x)
df = pd.get_dummies(df, columns=['Gender'])

df = df[(df['Age'] >= 10) & (df['Age'] <= 90)]

# Binary columns #10
binary_columns = ['self_employed', 'family_history', 'treatment', 'remote_work', 'tech_company']
for col in binary_columns:
    df[col] = df[col].str.strip().str.lower()
    df[col] = df[col].replace({
        'yes': 'Yes',
        'no': 'No',
        'na': np.nan,
        'n/a': np.nan,
    })
# Fill self_employed separately before mapping
df['self_employed'] = df['self_employed'].replace(r'^\s*$', np.nan, regex=True)
df['self_employed'] = df['self_employed'].fillna('No')

# Now encode all
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})


# Categorical columns #
df['work_interfere'] = df['work_interfere'].str.strip().str.lower()
df['work_interfere'] = df['work_interfere'].replace({
    "don't know": np.nan,
    'na': np.nan,
    'n/a': np.nan
})
df['work_interfere'] = df['work_interfere'].replace(r'^\s*$', np.nan, regex=True)
df['work_interfere'] = df['work_interfere'].fillna(df['work_interfere'].mode()[0])
df['work_interfere'] = df['work_interfere'].replace({
    'never': 'Never',
    'rarely': 'Rarely',
    'sometimes': 'Sometimes',
    'often': 'Often'
})
df['work_interfere'] = df['work_interfere'].map({'Often': 1, 'Never': 0, 'Sometimes':2, 'Rarely':3})
df['leave'] = df['leave'].str.strip().str.lower()
df['leave'] = df['leave'].replace({
    "very difficult": "Difficult",
    "somewhat difficult": "Difficult",
    "somewhat easy": "Easy",
    "very easy": "Easy",
    "don't know": np.nan,
    'na': np.nan
})
df['leave'] = df['leave'].replace(r'^\s*$', np.nan, regex=True)
df['leave'] = df['leave'].fillna(df['leave'].mode()[0])
df['leave'] = df['leave'].map({'Difficult': 1, 'Easy': 0})
# no of employees supportive #
df['no_employees'] = df['no_employees'].str.strip().str.lower()
df['no_employees'] = df['no_employees'].replace({
    'jun-25': np.nan,
    '01-may': np.nan,
    "don't know": np.nan,
    'n/a': np.nan,
    'na': np.nan
})
midpoint_map = {
    '1-5': 3,
    '6-25': 15,
    '26-100': 63,
    '100-500': 300,
    '500-1000': 750,
    'more than 1000': 1500
}
df['no_employees_mid'] = df['no_employees'].map(midpoint_map)
df['no_employees_mid'] = df['no_employees_mid'].fillna(df['no_employees_mid'].median())
df.drop(columns=['no_employees'], inplace=True)

# employer_related columns #
employer_columns = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity']
for col in employer_columns:
    df[col] = df[col].str.strip().str.lower()
    df[col] = df[col].replace({
        'yes': 'Yes',
        'no': 'No',
        "don't know": 'Unknown',
        'not sure': 'Unknown',
        'na': 'Unknown',
        'n/a': 'Unknown',
        np.nan: 'Unknown'
    })
    df[col] = df[col].fillna('Unknown')
for col in employer_columns:
    df[col] = df[col].map({'No': 0, 'Maybe': 2, 'Unknown': 2, 'Yes': 1})

# perception_based columns #
perception_columns = [
    'mental_health_interview', 'phys_health_interview',
    'mental_vs_physical', 'mental_health_consequence',
    'phys_health_consequence', 'obs_consequence'
]

for col in perception_columns:
    df[col] = df[col].str.strip().str.lower()
    df[col] = df[col].replace({
        'yes': 'Yes',
        'no': 'No',
        'maybe': 'Maybe',
        'not sure': 'Maybe',
        "don't know": 'Maybe',
        'na': 'Maybe',
        'n/a': 'Maybe'
    })
    df[col] = df[col].fillna('Maybe')
for col in perception_columns:
    df[col] = df[col].map({'No': 0, 'Maybe': 2, 'Yes': 1})

# communication related columns #
comm_columns = ['coworkers', 'supervisor']
for col in comm_columns:
    df[col] = df[col].str.strip().str.lower()
    df[col] = df[col].replace({
        'yes': 'Yes',
        'no': 'No',
        'some of them': 'Maybe',
        'maybe': 'Maybe',
        'not sure': 'Maybe',
        "don't know": 'Maybe',
        'na': 'Maybe',
        'n/a': 'Maybe'
    })
    df[col] = df[col].fillna('Maybe')
for col in comm_columns:
    df[col] = df[col].map({'No': 0, 'Maybe': 2, 'Yes': 1})

df.drop(columns=['comments','Timestamp'], inplace=True)
df['state'] = df['state'].replace(r'^\s*$', np.nan, regex=True)
df['state'] = df['state'].fillna('Unknown')

print("Missing in self_employed after mapping:", df['self_employed'].isna().sum())
print("Missing in treatment after mapping:", df['treatment'].isna().sum())
# List of columns with string categories to encode
cat_cols = df.select_dtypes(include='object').columns.tolist()
# Remove the target and optional columns from encoding
cat_cols = [col for col in cat_cols if col not in ['Country', 'state', 'treatment']]
df = pd.get_dummies(df, columns=cat_cols)

df.to_csv("cleaned_dataset.csv", index=False) 