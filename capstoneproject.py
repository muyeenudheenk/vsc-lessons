import pandas as pd
import numpy as np

# Load your CSVs (replace with your real file names)
gold = pd.read_csv('data/gold_prices.csv')
nifty = pd.read_csv('data/nifty50.csv')

# Calculate log returns
gold['log_return'] = np.log(gold['Price'] / gold['Price'].shift(1))
nifty['log_return'] = np.log(nifty['Price'] / nifty['Price'].shift(1))

# Drop missing values
gold.dropna(inplace=True)
nifty.dropna(inplace=True)

# Calculate correlation
corr = gold['log_return'].corr(nifty['log_return'])
print(f"Correlation between gold and NIFTY log returns: {corr:.4f}")

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
gold = pd.read_csv('data/gold_prices.csv')
nifty = pd.read_csv('data/nifty50.csv')

# Compute log returns
gold['log_return'] = np.log(gold['Price'] / gold['Price'].shift(1))
nifty['log_return'] = np.log(nifty['Price'] / nifty['Price'].shift(1))

# Align data
data = pd.concat([gold['log_return'], nifty['log_return']], axis=1).dropna()
data.columns = ['gold', 'nifty']

# Regression model
X = sm.add_constant(data['gold'])
model = sm.OLS(data['nifty'], X).fit()

print(model.summary())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
gold = pd.read_csv('data/gold_prices.csv')
nifty = pd.read_csv('data/nifty50.csv')

gold['log_return'] = np.log(gold['Price'] / gold['Price'].shift(1))
nifty['log_return'] = np.log(nifty['Price'] / nifty['Price'].shift(1))

# Plot
plt.figure(figsize=(10,5))
plt.plot(gold['Date'], gold['log_return'], label='Gold')
plt.plot(nifty['Date'], nifty['log_return'], label='NIFTY 50')
plt.title('Gold vs NIFTY Log Returns (2015–2024)')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()

