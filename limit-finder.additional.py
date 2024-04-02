import pandas as pd

# Read the CSV data into a DataFrame
df = pd.read_csv("crop_yield_data.csv")

# Find the highest and lowest values (numerical) for each column
for col in df.columns:
    # Check if the data type is numeric before finding min/max
    if pd.api.types.is_numeric_dtype(df[col]):
        highest = df[col].max()
        lowest = df[col].min()
        print(f"'{col}' : ({ lowest},  {highest }),")
    else:
        print(f"Column '{col}' is not numerical.")



#* Output-->

# 'N' : (15.0,  160.0),
# 'P' : (6.0,  100.0),
# 'K' : (12.0,  130.0),
# 'pH' : (3.8,  9.6),
# 'Humidity' : (32.0,  135.0),
# 'Temperature' : (4.0,  100.0),
# 'Rainfall' : (450.0,  4150.0),
# 'CropYield' : (8.0,  8800.0),
# Column 'District' is not numerical.
# Column 'SoilType' is not numerical.
# Column 'CropName' is not numerical.