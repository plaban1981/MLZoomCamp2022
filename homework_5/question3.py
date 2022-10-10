import joblib

def load(filename):
        return joblib.load(filename)


dv = load('dv.bin')
model = load('model1.bin')

customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)