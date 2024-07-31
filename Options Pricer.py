import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def fetch_and_prepare_data(ticker, num_dates=5):
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options[:num_dates]
    options_data = []

    for date in expiration_dates:
        options = stock.option_chain(date)
        calls = options.calls.assign(option_type='call')
        puts = options.puts.assign(option_type='put')
        options_data.append(pd.concat([calls, puts]))

    df = pd.concat(options_data)
    current_time = pd.Timestamp.now(tz='UTC')
    df['time_to_expiration'] = (pd.to_datetime(df['lastTradeDate']) - current_time).dt.total_seconds() / (
                365 * 24 * 60 * 60)

    return df, expiration_dates


def prepare_features_and_target(df):
    features = ['strike', 'time_to_expiration', 'impliedVolatility', 'volume', 'openInterest']
    X = df[features]
    y = df['lastPrice']

    feature_imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(feature_imputer.fit_transform(X), columns=X.columns)
    y_imputed = pd.Series(SimpleImputer(strategy='mean').fit_transform(y.values.reshape(-1, 1)).ravel(),
                          name='lastPrice')

    return X_imputed, y_imputed, feature_imputer, features


def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, scaler, mse, r2, y_test, y_pred


def predict_future_prices(model, scaler, feature_imputer, features, stock, expiration_date, current_time):
    future_options = stock.option_chain(expiration_date)
    future_options_df = pd.concat([future_options.calls, future_options.puts])
    future_options_df['time_to_expiration'] = (pd.to_datetime(expiration_date) - current_time).total_seconds() / (
                365 * 24 * 60 * 60)

    future_X = future_options_df[features]
    future_X_imputed = pd.DataFrame(feature_imputer.transform(future_X), columns=features)
    future_X_scaled = scaler.transform(future_X_imputed)

    future_options_df['predicted_price'] = model.predict(future_X_scaled)
    return future_options_df[['strike', 'lastPrice', 'predicted_price']]


# Main execution
df, expiration_dates = fetch_and_prepare_data("QQQ")
X, y, feature_imputer, features = prepare_features_and_target(df)
model, scaler, mse, r2, y_test, y_pred = train_and_evaluate_model(X, y)

print(f"Total options data shape: {df.shape}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Option Prices")
plt.show()

if len(expiration_dates) > 5:
    future_predictions = predict_future_prices(model, scaler, feature_imputer, features, yf.Ticker("QQQ"),
                                               expiration_dates[5], pd.Timestamp.now(tz='UTC'))
    print("\nFuture Price Predictions:")
    print(future_predictions)