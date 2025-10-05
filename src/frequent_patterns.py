import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Load dataset
df = pd.read_csv("../data/processed/S&P500_merged.csv", parse_dates=["Date"])

# --- Step 1: Discretize continuous variables ---
df_bin = pd.DataFrame()
df_bin["Close_High"] = df["Close"] > df["Close"].median()
df_bin["M2_High"] = df["M2_Money_Supply"] > df["M2_Money_Supply"].median()
df_bin["GDP_High"] = df["GDP"] > df["GDP"].median()
df_bin["Unemployment_Low"] = df["Unemployment"] < df["Unemployment"].median()
df_bin["CPI_High"] = df["CPI"] > df["CPI"].median()
df_bin["Volume_High"] = df["Volume"] > df["Volume"].median()

df_bool = df_bin.astype(bool)

# --- Step 2: Print label distributions ---
print("\n✅ Binary Column True Ratios:")
print(df_bool.mean())

# --- Step 3: Apriori ---
frequent_apriori = apriori(df_bool, min_support=0.2, use_colnames=True)
print(f"\n🔎 Apriori found {len(frequent_apriori)} frequent itemsets")

if not frequent_apriori.empty:
    rules_apriori = association_rules(frequent_apriori, metric="lift", min_threshold=1)
    rules_apriori.to_csv("../outputs/rules_apriori.csv", index=False)
    print(f"✅ Apriori generated {len(rules_apriori)} rules")
else:
    print("❌ No frequent itemsets found with Apriori")

# --- Step 4: FP-Growth ---
frequent_fpgrowth = fpgrowth(df_bool, min_support=0.2, use_colnames=True)
print(f"\n🔎 FP-Growth found {len(frequent_fpgrowth)} frequent itemsets")

if not frequent_fpgrowth.empty:
    rules_fpgrowth = association_rules(frequent_fpgrowth, metric="lift", min_threshold=1)
    rules_fpgrowth.to_csv("../outputs/rules_fpgrowth.csv", index=False)
    print(f"✅ FP-Growth generated {len(rules_fpgrowth)} rules")
else:
    print("❌ No frequent itemsets found with FP-Growth")