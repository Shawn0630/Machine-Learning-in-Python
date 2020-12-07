from sklearn import preprocessing


class RawDataUtilities:
    @staticmethod
    def normalize_feature(x_training_mat):
        return preprocessing.scale(x_training_mat)

    @staticmethod
    def polynomial_features(x_training_mat, degree):
        poly = preprocessing.PolynomialFeatures(degree=degree)
        return poly.fit_transform(x_training_mat)

    @staticmethod
    def normalize_feature_data_frame(df):
        """Applies function along input axis(default 0) of DataFrame.

        Parameters
        ----------
        df: DataFrame

        """
        return df.apply(lambda column: (column - column.mean()) / column.std())
