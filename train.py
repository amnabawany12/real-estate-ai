import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
locs = ['DHA Phase 6','Bahria Town','Gulshan-e-Iqbal','Clifton','PECHS','Johar Town','Model Town']
df = pd.DataFrame({
    'Location': np.random.choice(locs, 500),
    'Bedrooms': np.random.choice([1,2,3,4,5], 500),
    'Bathrooms': np.random.choice([1,2,3,4], 500),
    'Area_sqft': np.random.randint(600,4000,500),
    'Age_years': np.random.randint(0,30,500),
    'Furnished': np.random.choice([0,1], 500)
})
mult = {'DHA Phase 6':1.8,'Bahria Town':1.5,'Clifton':2.0,'Gulshan-e-Iqbal':1.0,'PECHS':1.3,'Johar Town':1.1,'Model Town':1.2}
df['Price_PKR'] = df.apply(lambda r: int((r.Area_sqft*15000 + r.Bedrooms*2000000 - r.Age_years*100000)*mult[r.Location]), axis=1)
df['Price_PKR'] = df['Price_PKR'].apply(lambda x: max(x, 3000000))
df.to_csv('real_estate.csv', index=False)

X = df[['Location','Bedrooms','Bathrooms','Area_sqft','Age_years','Furnished']]
y = df['Price_PKR']
pre = ColumnTransformer([('loc', OneHotEncoder(handle_unknown='ignore'), ['Location'])], remainder='passthrough')
model = Pipeline([('pre',pre),('rf',RandomForestRegressor(100, random_state=42))])
model.fit(X,y)
joblib.dump(model,'model.pkl')
print("DONE! Dataset and model created")