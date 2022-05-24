from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, load_data, clean_data
from TaxiFareModel.utils import compute_rmse

class Trainer():


    def __init__(self, X, y):
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        pipe = Pipeline([
        ('preproc', preproc_pipe),
        ('linear_model', LinearRegression())
        ])

        self.pipeline = pipe

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 2)



if __name__ == "__main__":
    df = load_data(n_rows=10000)
    df = clean_data(df)
    y = df['fare_amount']
    X = df.drop('fare_amount', axis=1)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
