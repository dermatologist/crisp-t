import logging
from typing import Optional

import numpy as np
import pandas as pd

from .model import Corpus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Csv:

    def __init__(
        self,
        corpus: Optional[Corpus] = None,
        comma_separated_text_columns: str = "",
        comma_separated_ignore_columns: str = "",
        id_column: str = "id",
    ):
        """
        Initialize the Csv object.
        """
        self._corpus = corpus
        if self._corpus is None:
            self._df = pd.DataFrame()
            logger.info("No corpus provided. Creating an empty DataFrame.")
        else:
            self._df = self._corpus.df
            if self._df is None:
                logger.info("No DataFrame found in the corpus. Creating a new one.")
                self._df = pd.DataFrame()
        self._df_original = self._df.copy()
        self._comma_separated_text_columns = comma_separated_text_columns
        self._comma_separated_ignore_columns = comma_separated_ignore_columns
        self._id_column = id_column
        self._X = None
        self._y = None
        self._X_original = None
        self._y_original = None

    @property
    def corpus(self) -> Optional[Corpus]:
        if self._corpus is not None and self._df is not None:
            self._corpus.df = self._df
        return self._corpus

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            return pd.DataFrame()
        return self._df

    @property
    def comma_separated_text_columns(self) -> str:
        return self._comma_separated_text_columns

    @property
    def comma_separated_ignore_columns(self) -> str:
        return self._comma_separated_ignore_columns

    @comma_separated_ignore_columns.setter
    def comma_separated_ignore_columns(self, value: str) -> None:
        self._comma_separated_ignore_columns = value
        logger.info("Comma-separated ignore columns set successfully.")
        logger.debug(
            f"Comma-separated ignore columns: {self._comma_separated_ignore_columns}"
        )

    @property
    def id_column(self) -> str:
        return self._id_column

    @corpus.setter
    def corpus(self, value: Corpus) -> None:
        self._corpus = value
        if self._corpus is not None:
            self._df = self._corpus.df
            if self._df is None:
                logger.info("No DataFrame found in the corpus. Creating a new one.")
                self._df = pd.DataFrame()
            self._df_original = self._df.copy()
            logger.info("Corpus set successfully.")
            logger.debug(f"DataFrame content: {self._df.head()}")
            logger.debug(f"DataFrame shape: {self._df.shape}")
            logger.debug(f"DataFrame columns: {self._df.columns.tolist()}")
        else:
            logger.error("Failed to set corpus. Corpus is None.")

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        self._df = value
        logger.info("DataFrame set successfully.")
        logger.debug(f"DataFrame content: {self._df.head()}")
        logger.debug(f"DataFrame shape: {self._df.shape}")
        logger.debug(f"DataFrame columns: {self._df.columns.tolist()}")

    @comma_separated_text_columns.setter
    def comma_separated_text_columns(self, value: str) -> None:
        self._comma_separated_text_columns = value
        logger.info("Comma-separated text columns set successfully.")
        logger.debug(
            f"Comma-separated text columns: {self._comma_separated_text_columns}"
        )

    @id_column.setter
    def id_column(self, value: str) -> None:
        self._id_column = value
        logger.info("ID column set successfully.")
        logger.debug(f"ID column: {self._id_column}")

    # TODO remove @deprecated
    #! Do not use
    def read_csv(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV file and create a DataFrame.
        """
        try:
            self._df = pd.read_csv(file_path)
            logger.info(f"CSV file {file_path} read successfully.")
            logger.debug(f"DataFrame content: {self._df.head()}")
            logger.debug(f"DataFrame shape: {self._df.shape}")
            logger.debug(f"DataFrame columns: {self._df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        # ignore comma-separated text columns
        if self._comma_separated_text_columns:
            text_columns = [
                col.strip()
                for col in self._comma_separated_text_columns.split(",")
                if col.strip()
            ]
            for col in text_columns:
                if col in self._df.columns:
                    self._df[col] = self._df[col].astype(str)
                    logger.info(f"Column {col} converted to string.")
                    logger.debug(f"Column {col} content: {self._df[col].head()}")
                else:
                    logger.warning(f"Column {col} not found in DataFrame.")
        return self._df

    def write_csv(self, file_path: str, index: bool = False) -> None:
        if self._df is not None:
            self._df.to_csv(file_path, index=index)
            logger.info(f"DataFrame written to {file_path}")
            logger.debug(f"DataFrame content: {self._df.head()}")
            logger.debug(f"Index: {index}")
        else:
            logger.error("DataFrame is None. Cannot write to CSV.")

    def mark_missing(self):
        if self._df is not None:
            self._df.replace("", np.nan, inplace=True)
            self._df.dropna(inplace=True)
        else:
            logger.error("DataFrame is None. Cannot mark missing values.")

    def mark_duplicates(self):
        if self._df is not None:
            self._df.drop_duplicates(inplace=True)
        else:
            logger.error("DataFrame is None. Cannot mark duplicates.")

    def restore_df(self):
        self._df = self._df_original.copy()

    def get_shape(self):
        if self._df is not None:
            return self._df.shape
        else:
            logger.error("DataFrame is None. Cannot get shape.")
            return None

    def get_columns(self):
        if self._df is not None:
            return self._df.columns.tolist()
        else:
            logger.error("DataFrame is None. Cannot get columns.")
            return []

    def get_column_types(self):
        if self._df is not None:
            return self._df.dtypes.to_dict()
        else:
            logger.error("DataFrame is None. Cannot get column types.")
            return {}

    def get_column_values(self, column_name: str):
        if self._df is not None and column_name in self._df.columns:
            return self._df[column_name].tolist()
        else:
            logger.error(
                f"Column {column_name} not found in DataFrame or DataFrame is None."
            )
            return None

    def read_xy(self, y: str, ignore_columns=True, numeric_only=False, filter_nans=True, comma_separated_include_columns: str = ""):
        """
        Read X and y variables from the DataFrame.
        """
        if self._df is None:
            logger.error("DataFrame is None. Cannot read X and y.")
            return None, None
        if comma_separated_include_columns != "":
            include_cols = [
                col
                for col in comma_separated_include_columns.split(",")
                if col.strip() and col in self._df.columns
            ]
            self._df = self._df[include_cols + [y]] if y in self._df.columns else self._df[include_cols]
            logger.info(f"DataFrame filtered to include columns: {include_cols + [y] if y in self._df.columns else include_cols}")
        if numeric_only:
            self._df = self._df.select_dtypes(include=[np.number])
            logger.info("DataFrame filtered to numeric columns only.")
        if filter_nans:
            self._df = self._df.dropna()
            logger.info("Rows with NaN values dropped from DataFrame.")
        if y == "":
            self._y = None
        else:
            self._y = self._df[y]
        ignore_cols = [
            col
            for col in self._comma_separated_ignore_columns.split(",")
            if col.strip()
        ]
        if y != "":
            if ignore_columns and ignore_cols:
                self._X = self._df.drop(columns=[y] + ignore_cols)
            else:
                self._X = self._df.drop(columns=[y])
        else:
            if ignore_columns and ignore_cols:
                self._X = self._df.drop(columns=ignore_cols)
            else:
                self._X = self._df.copy()
        logger.info(f"X and y variables set. X shape: {self._X.shape}")
        return self._X, self._y

    def drop_na(self):
        if self._df is not None:
            self._df.dropna(inplace=True)
            logger.info("Missing values dropped from DataFrame.")
        else:
            logger.error("DataFrame is None. Cannot drop missing values.")

    def oversample(self):
        self._X_original = self._X
        self._y_original = self._y
        try:
            from imblearn.over_sampling import RandomOverSampler

            ros = RandomOverSampler(random_state=0)
        except ImportError:
            logger.info(
                "ML dependencies are not installed. Please install them by ```pip install crisp-t[ml] to use ML features."
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
        return X, y

    def restore_oversample(self):
        self._X = self._X_original
        self._y = self._y_original

    def prepare_data(self, y: str, oversample=False, one_hot_encode_all=False):
        self.mark_missing()
        if oversample:
            self.oversample()
        self.one_hot_encode_strings_in_df()
        if one_hot_encode_all:
            self.one_hot_encode_all_columns()
        return self.read_xy(y)

    def one_hot_encode_strings_in_df(self, n=10, filter_high_cardinality=False):
        if self._df is not None:
            categorical_cols = self._df.select_dtypes(include=["object"]).columns.tolist()
            # Remove categorical columns with more than n unique values
            if filter_high_cardinality:
                categorical_cols = [
                    col for col in categorical_cols if self._df[col].nunique() <= n
                ]
            if categorical_cols:
                self._df = pd.get_dummies(self._df, columns=categorical_cols, drop_first=True)
                logger.info("One-hot encoding applied to string columns.")
            else:
                logger.info("No string (object) columns found for one-hot encoding.")
        else:
            logger.error("DataFrame is None. Cannot apply one-hot encoding.")

    def one_hot_encode_all_columns(self):
        # The allowed values for a DataFrame are True, False, 0, 1. Found value 2
        # Map all values to 0 or 1 for one-hot encoding:
        # - 1 or True -> 1
        # - 0 or False -> 0
        # - Any other value -> 1 (with a warning)
        if self._df is not None:

            def to_one_hot(x):
                if x in [1, True]:
                    return True
                elif x in [0, False]:
                    return False
                else:
                    # logger.warning(
                    #     f"Unexpected value '{x}' encountered during one-hot encoding; mapping to 1."
                    # )
                    return True

            self._df = self._df.applymap(to_one_hot) # type: ignore
