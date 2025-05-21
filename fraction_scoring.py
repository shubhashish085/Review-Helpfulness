import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("final_merged.csv")

df = df.rename(columns={'apartmentName_x': 'apartmentName'})
df = df.drop(['apartmentName_y'], axis = 1)
df = df.drop(['Unnamed: 0'], axis = 1)

# Step 1: Replace NaN with "0 People Found This Helpful"
df['helpful'] = df['helpful'].fillna("0 People Found This Helpful")

# Step 2: Extract integer from the 'helpful' column
df['helpfulness'] = df['helpful'].str.extract(r'(\d+)').astype(int)

# Step 3: Normalize month names (e.g., "Sept." → "Sep.")
df['date'] = df['date'].str.replace("Sept", "Sep", regex=False)
# Step 3: Normalize month names (e.g., "Sept." → "Sep.")
df['date'] = df['date'].str.replace("July", "Jul", regex=False)
## Step 3: Normalize month names (e.g., "Sept." → "Sep.")
df['date'] = df['date'].str.replace("June", "Jun", regex=False)
df['date'] = df['date'].str.replace(".", "", regex=False)

# Step 4: Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y')

# Step 5: Calculate days since review
today = pd.Timestamp(datetime.today().date())
df['days_since_review'] = (today - df['date']).dt.days


df['apartment_loc'] = df['url'].str.split('/').apply(lambda x:x[3])
df['state'] = df['apartment_loc'].str.split('-').apply(lambda x:x[-1].upper())
df['city'] = df['apartment_loc'].str.split('-').apply(lambda x:x[-2].upper())

'''for i in range(len(df)):
    if(pd.isna(df.at[i, 'apartmentName'])):
        df.at[i, 'apartmentName'] = df['apartment_loc'].str.split('-').apply(lambda x:'-'.join(x[:len(x)-2]))'''

df.dropna(subset=['apartmentName'], inplace=True)

df = df.drop(['title', 'date'], axis = 1)
df = df.drop(['apartment_loc', 'helpful', 'response'], axis = 1)

group_max_helpfulness = df.groupby(['state', 'city', 'apartmentName'])['helpfulness'].max()

for i in range(len(df)):
    if(i in df.index):
        print(df.at[i, 'state'] ,',', df.at[i, 'city'] ,',', df.at[i, 'apartmentName'])
        max_helpfulness_for_aptmt = group_max_helpfulness.loc[df.at[i, 'state'], df.at[i, 'city'], df.at[i, 'apartmentName']]
        if(max_helpfulness_for_aptmt == 0):
            continue
        filtered_df = df[(df['state'] == df.at[i, 'state']) & (df['city'] == df.at[i, 'city']) & (df['apartmentName'] == df.at[i, 'apartmentName']) & (df['helpfulness'] == max_helpfulness_for_aptmt)]
        max_days_for_ref = filtered_df['days_since_review'].max()    
        if(df.at[i, 'days_since_review'] >=  max_days_for_ref):
            df.at[i, 'calculated_score'] = float(df.at[i, 'helpfulness'])/ float(max_helpfulness_for_aptmt)
        else:
            df.at[i, 'calculated_score'] = min((float(df.at[i, 'helpfulness'] * max_days_for_ref) / float(max_helpfulness_for_aptmt * df.at[i, 'days_since_review'])), 1.0)

df.to_csv('scored_helpfulness_data.csv')
