import logging
import warnings

import numpy as np
import pandas as pd

from .model import Corpus

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Csv:

    def __init__(
        self,
        corpus: Corpus | None = None,
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
        self._id_column = id_column

    @property
    def corpus(self) -> Corpus | None:
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
        self._process_columns()

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
        self._process_columns()

    @id_column.setter
    def id_column(self, value: str) -> None:
        self._id_column = value
        # Add id column to the list of ignored columns
        ignore_cols = [
            col
            for col in self._comma_separated_ignore_columns.split(",")
            if col.strip()
        ]
        if value not in ignore_cols:
            ignore_cols.append(value)
            self._comma_separated_ignore_columns = ",".join(ignore_cols)
            logger.debug(
                f"ID column '{value}' added to ignore columns: {self._comma_separated_ignore_columns}"
            )
        logger.info("ID column set successfully.")
        logger.debug(f"ID column: {self._id_column}")

    # TODO remove @deprecated
    #! Do not use
    def read_csv(self, file_path: str):
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
            logger.exception(f"Error reading CSV file: {e}")
            raise
        return self._process_columns()

    def _process_columns(self):
        # ignore comma-separated ignore columns
        if self._comma_separated_ignore_columns:
            ignore_columns = [
                col.strip()
                for col in self._comma_separated_ignore_columns.split(",")
                if col.strip()
            ]
            self._df.drop(columns=ignore_columns, inplace=True, errors="ignore")
            logger.info(
                f"Ignored columns: {ignore_columns}. Updated DataFrame shape: {self._df.shape}"
            )
            logger.debug(f"DataFrame content after dropping columns: {self._df.head()}")
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
        # ignore all columns with names starting with "metadata_"
        self._df = self._df.loc[:, ~self._df.columns.str.startswith("metadata_")]
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
        """Mark missing values in the DataFrame.
        Missing values are considered as empty strings and are replaced with NaN.
        Rows with NaN values are then dropped from the DataFrame.
        """
        if self._df is not None:
            self._df.replace("", np.nan, inplace=True)
            self._df.dropna(inplace=True)
        else:
            logger.error("DataFrame is None. Cannot mark missing values.")

    def mark_duplicates(self):
        """Mark duplicate rows in the DataFrame.
        Duplicate rows are identified and dropped from the DataFrame.
        """
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
        """Get the list of columns in the DataFrame."""
        if self._df is not None:
            return self._df.columns.tolist()
        else:
            logger.error("DataFrame is None. Cannot get columns.")
            return []

    def get_column_types(self):
        """Get the data types of columns in the DataFrame."""
        if self._df is not None:
            return self._df.dtypes.to_dict()
        else:
            logger.error("DataFrame is None. Cannot get column types.")
            return {}

    def get_column_values(self, column_name: str):
        """Get the unique values in a column of the DataFrame."""
        if self._df is not None and column_name in self._df.columns:
            return self._df[column_name].tolist()
        else:
            logger.error(
                f"Column {column_name} not found in DataFrame or DataFrame is None."
            )
            return None

    def retain_numeric_columns_only(self):
        """Retain only numeric columns in the DataFrame."""
        if self._df is not None:
            self._df = self._df.select_dtypes(include=[np.number])
            logger.info("DataFrame filtered to numeric columns only.")
        else:
            logger.error("DataFrame is None. Cannot filter to numeric columns.")

    def comma_separated_include_columns(self, include_cols: str = ""):
        """Retain only specified columns in the DataFrame."""
        if include_cols == "":
            return
        if self._df is not None:
            cols = [
                col.strip()
                for col in include_cols.split(",")
                if col.strip() and col in self._df.columns
            ]
            self._df = self._df[cols]
            logger.info(f"DataFrame filtered to include columns: {cols}")
        else:
            logger.error("DataFrame is None. Cannot filter to include columns.")

    def read_xy(self, y: str):
        """
        Read X and y variables from the DataFrame.
        """
        if self._df is None:
            logger.error("DataFrame is None. Cannot read X and y.")
            return None, None
        # Split into X and y
        if y == "":
            self._y = None
        else:
            self._y = self._df[y]
        if y != "":
            self._X = self._df.drop(columns=[y])
        else:
            self._X = self._df.copy()
        logger.info(f"X and y variables set. X shape: {self._X.shape}")
        return self._X, self._y

    def drop_na(self):
        """Drop rows with any NA values from the DataFrame."""
        if self._df is not None:
            self._df.dropna(inplace=True)
            logger.info("Missing values dropped from DataFrame.")
        else:
            logger.error("DataFrame is None. Cannot drop missing values.")

    def oversample(self, mcp: bool = False):
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
        if mcp:
            return f"Oversampling completed. New X shape: {self._X.shape}"
        return X, y

    def restore_oversample(self, mcp: bool = False):
        self._X = self._X_original
        self._y = self._y_original
        if mcp:
            return f"Oversampling restored. X shape: {self._X.shape}, y shape: {self._y.shape}"  # type: ignore

    def prepare_data(self, y: str, oversample=False, one_hot_encode_all=False):
        self.mark_missing()
        if oversample:
            self.oversample()
        self.one_hot_encode_strings_in_df()
        if one_hot_encode_all:
            self.one_hot_encode_all_columns()
        return self.read_xy(y)

    def bin_a_column(self, column_name: str, bins: int = 2):
        """Bin a numeric column into specified number of bins."""
        if self._df is not None and column_name in self._df.columns:
            if pd.api.types.is_numeric_dtype(self._df[column_name]):
                self._df[column_name] = pd.cut(
                    self._df[column_name], bins=bins, labels=False
                )
                logger.info(f"Column {column_name} binned into {bins} bins.")
                return "I have binned the column. Please proceed."
            else:
                logger.warning(f"Column {column_name} is not numeric. Cannot bin.")
        else:
            logger.warning(
                f"Column {column_name} not found in DataFrame or DataFrame is None."
            )
        return "I cannot bin the column. Please check the logs for more information."

    def one_hot_encode_column(self, column_name: str):
        """One-hot encode a specific column in the DataFrame.
        This method converts a categorical column into one-hot encoded columns.
        Used when # ValueError: could not convert string to float.
        """
        if self._df is not None and column_name in self._df.columns:
            if pd.api.types.is_object_dtype(self._df[column_name]):
                self._df = pd.get_dummies(
                    self._df, columns=[column_name], drop_first=True
                )
                logger.info(f"One-hot encoding applied to column {column_name}.")
                return "I have one-hot encoded the column. Please proceed."
            else:
                logger.warning(f"Column {column_name} is not of object type.")
        else:
            logger.error(
                f"Column {column_name} not found in DataFrame or DataFrame is None."
            )
        return "I cannot one-hot encode the column. Please check the logs for more information."

    def one_hot_encode_strings_in_df(self, n=10, filter_high_cardinality=False):
        """One-hot encode string (object) columns in the DataFrame.
        This method converts categorical string columns into one-hot encoded columns.
        Columns with more than n unique values can be optionally filtered out.
        Used when # ValueError: could not convert string to float.
        """
        if self._df is not None:
            categorical_cols = self._df.select_dtypes(
                include=["object"]
            ).columns.tolist()
            # Remove categorical columns with more than n unique values
            if filter_high_cardinality:
                categorical_cols = [
                    col for col in categorical_cols if self._df[col].nunique() <= n
                ]
            if categorical_cols:
                self._df = pd.get_dummies(
                    self._df, columns=categorical_cols, drop_first=True
                )
                logger.info("One-hot encoding applied to string columns.")
            else:
                logger.info("No string (object) columns found for one-hot encoding.")
        else:
            logger.error("DataFrame is None. Cannot apply one-hot encoding.")

    def one_hot_encode_all_columns(self):
        """One-hot encode all columns in the DataFrame.
        This method converts all values in the DataFrame to boolean values.
        Used for apriori algorithm which requires boolean values.
        """
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

            self._df = self._df.applymap(to_one_hot)  # type: ignore

    def filter_rows_by_column_value(
        self, column_name: str, value, mcp: bool = False
    ):
        """Select rows from the DataFrame where the specified column matches the given value.
        Additionally, filter self._corpus.documents by id_column if present in DataFrame.
        """
        if self._df is not None and column_name in self._df.columns:
            selected_df = self._df[self._df[column_name] == value]
            if selected_df.empty:
                # try int search
                try:
                    selected_df = self._df[self._df[column_name] == int(value)]
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert value '{value}' to int for column '{column_name}'."
                    )
            logger.info(
                f"Selected {selected_df.shape[0]} rows where {column_name} == {value}."
            )
            self._df = selected_df

            # Check for id_column in DataFrame
            if (
                self._corpus is not None
                and hasattr(self._corpus, "df")
                and self._id_column in self._corpus.df.columns
            ):
                logger.info(f"id_column '{self._id_column}' exists in DataFrame.")
                valid_ids = set(self._corpus.df[self._id_column].tolist())
                if (
                    hasattr(self._corpus, "documents")
                    and self._corpus.documents is not None
                ):
                    filtered_docs = [
                        doc
                        for doc in self._corpus.documents
                        if getattr(doc, self._id_column, None) in valid_ids
                    ]
                    self._corpus.documents = filtered_docs
            else:
                logger.warning(f"id_column '{self._id_column}' does not exist in DataFrame.")

            if mcp:
                return f"Selected {selected_df.shape[0]} rows where {column_name} == {value}."
        else:
            logger.warning(
                f"Column {column_name} not found in DataFrame or DataFrame is None."
            )
            if mcp:
                return (
                    f"Column {column_name} not found in DataFrame or DataFrame is None."
                )
            return pd.DataFrame()
