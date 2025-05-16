import pandas as pd
import logging
import numpy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Csv:

    def __init__(
        self,
        df: pd.DataFrame = pd.DataFrame(),
        comma_separated_text_columns: str = "",
        id_column: str = "id",
    ):
        """
        Initialize the Csv object.
        """
        self._df = df
        self._df_original = df.copy()
        self._comma_separated_text_columns = comma_separated_text_columns
        self._id_column = id_column
        self._X = None
        self._y = None
        self._X_original = None
        self._y_original = None

    @property
    def df(self) -> pd.DataFrame:
        """
        Get the DataFrame.
        """
        return self._df

    @property
    def comma_separated_text_columns(self) -> str:
        """
        Get the comma-separated text columns.
        """
        return self._comma_separated_text_columns

    @property
    def id_column(self) -> str:
        """
        Get the ID column.
        """
        return self._id_column

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        """
        Set the DataFrame.
        """
        self._df = value
        logger.info("DataFrame set successfully.")
        logger.debug(f"DataFrame content: {self._df.head()}")
        logger.debug(f"DataFrame shape: {self._df.shape}")
        logger.debug(f"DataFrame columns: {self._df.columns.tolist()}")

    @comma_separated_text_columns.setter
    def comma_separated_text_columns(self, value: str) -> None:
        """
        Set the comma-separated text columns.
        """
        self._comma_separated_text_columns = value
        logger.info("Comma-separated text columns set successfully.")
        logger.debug(
            f"Comma-separated text columns: {self._comma_separated_text_columns}"
        )

    @id_column.setter
    def id_column(self, value: str) -> None:
        """
        Set the ID column.
        """
        self._id_column = value
        logger.info("ID column set successfully.")
        logger.debug(f"ID column: {self._id_column}")

    def write_csv(self, file_path: str, index: bool = False) -> None:
        """
        Write the DataFrame to a CSV file.
        """
        self._df.to_csv(file_path, index=index)
        logger.info(f"DataFrame written to {file_path}")
        logger.debug(f"DataFrame content: {self._df.head()}")
        logger.debug(f"Index: {index}")

    def mark_missing(self):
        self._df = self._df.replace("", numpy.nan)
        self._df.dropna(inplace=True)

    def restore_mark_missing(self):
        self._df = self._df_original.copy()

    def get_shape(self):
        return self._df.shape

    def get_columns(self):
        return self._df.columns.tolist()

    def get_column_types(self):
        return self._df.dtypes.to_dict()

    def get_column_values(self, column_name: str):
        if column_name in self._df.columns:
            return self._df[column_name].tolist()
        else:
            logger.error(f"Column {column_name} not found in DataFrame.")
            return None

    def read_xy(self, y: str):
        self._y = self._df[y]
        self._X = self._df.drop(columns=[y])
        logger.info(
            f"X and y variables set. X shape: {self._X.shape}, y shape: {self._y.shape}"
        )
        return self._X, self._y

    def oversample(self):
        self._X_original = self._X
        self._y_original = self._y
        try:
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=0)
        except ImportError:
            logger.info(
                "ML dependencies are not installed. Please install them by ```pip install qrmine[ml] to use ML features."
            )
            return
        result = ros.fit_resample(self._X, self._y)
        if len(result) == 2:
            X, y = result
        elif len(result) == 3:
            X, y, _ = result
        else:
            logger.error("Unexpected number of values returned from fit_resample.")
            return
        self._X = X
        self._y = y

    def restore_oversample(self):
        self._X = self._X_original
        self._y = self._y_original
