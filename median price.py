import pandas as pd
import matplotlib.pyplot as plt

# Load your Airbnb listings data
listings_df = pd.read_csv('listings.csv')

# Clean the 'price' column and convert it to numeric
listings_df['price'] = listings_df['price'].replace('[\$,]', '', regex=True).astype(float)

# Group by neighborhood and calculate the median price
median_price_by_neighborhood = listings_df.groupby('neighbourhood')['price'].median().sort_values(ascending=False)

# Visualize the results
plt.figure(figsize=(12, 6))

# Bar plot for median price
plt.bar(median_price_by_neighborhood.index, median_price_by_neighborhood, color='salmon')
plt.title('Median Price in Each Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Median Price')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
