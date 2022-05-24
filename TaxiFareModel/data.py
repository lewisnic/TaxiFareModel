import pandas as pd

def get_data(nrows=10000):
    '''returns a DataFrame with nrows from s3 bucket'''
    aws_path = "https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-train.csv"
    df = pd.read_csv(aws_path, nrows=nrows)
    return df

def load_data(n_rows=1_000_000):
    url = 's3://wagon-public-datasets/taxi-fare-train.csv'
    local_path = './raw_data/train.csv'

    try:
        print("Loading data online.")
        df = pd.read_csv(url, nrows=n_rows)

    except:
        print("Loading data online FAILED.")
        print("Loading data locally!")
        df = pd.read_csv(local_path, nrows=n_rows)

    return df


def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df.dropna(how='any')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

if __name__ == "__main__":
    print(load_data(n_rows=10))
