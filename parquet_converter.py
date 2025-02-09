import pandas as pd

# Load the Parquet file
df = pd.read_parquet("static/BTC_USDT_all_tf_merged.parquet")

# Take the first 10,000 rows and the last 10,000 rows
df_head = df.head(16000)
df_tail = df.tail(100)

# Combine them
df_combined = pd.concat([df_head, df_tail])

# Save to CSV
df_combined.to_csv("static/ETH_USDT_all_tf_merged_limited.csv", index=False)
