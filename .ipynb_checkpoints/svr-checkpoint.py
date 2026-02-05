



from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# 1. Create simple fake data (or you can use your own)
X, y = make_regression(n_samples=300, n_features=1, noise=20, random_state=42)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# 3. Scale (very important!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4. Create and train SVR
model = SVR(kernel='rbf', C=10, epsilon=0.5)
model.fit(X_train, y_train)

# 5. Predict and check error
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print(f"Average error (MAE): {error:.2f}")