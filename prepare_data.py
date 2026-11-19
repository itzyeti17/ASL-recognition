from data_pipeline import load_dataset, split_dataset


X, y = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst normalized example:")
print(X[0])

print("\nCorrect answer:")
print(y[0])

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
