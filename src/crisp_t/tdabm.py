"""
Copyright (C) 2025 Bell Eapen

This file is part of crisp-t.

crisp-t is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

crisp-t is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with crisp-t.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import random

import numpy as np
import pandas as pd

from .model import Corpus

logger = logging.getLogger(__name__)


class Tdabm:
    """
    Topological Data Analysis Ball Mapper (TDABM) implementation.

    Based on the algorithm by Rudkin and Dlotko (2024), this class provides
    a model-free means to visualize multidimensional data using ball mapping.

    The TDABM algorithm creates a point cloud from multidimensional data and
    covers it with overlapping balls. Balls are connected if they have non-empty
    intersections, creating a topological representation of the data structure.
    """

    def __init__(self, corpus: Corpus):
        """
        Initialize the TDABM analyzer.

        Args:
            corpus: Corpus object containing the data to analyze
        """
        self.corpus = corpus
        if self.corpus.df is None:
            raise ValueError("Corpus must have a DataFrame for TDABM analysis")

    def _validate_variables(self, y: str, x_variables: list[str]) -> None:
        """
        Validate that y is continuous and X variables are ordinal.

        Args:
            y: Name of the y variable (continuous)
            x_variables: List of X variable names (ordinal)

        Raises:
            ValueError: If variables are invalid
        """
        df = self.corpus.df

        if y not in df.columns:
            raise ValueError(f"Y variable '{y}' not found in DataFrame columns")

        # Check y is numeric and continuous (not binary)
        if not pd.api.types.is_numeric_dtype(df[y]):
            raise ValueError(f"Y variable '{y}' must be numeric/continuous")

        # Check if y appears to be binary
        unique_vals = df[y].dropna().nunique()
        if unique_vals == 2:
            raise ValueError(f"Y variable '{y}' appears to be binary (only 2 unique values). TDABM requires a continuous variable.")

        # Check X variables exist and are numeric
        for x in x_variables:
            if x not in df.columns:
                raise ValueError(f"X variable '{x}' not found in DataFrame columns")
            if not pd.api.types.is_numeric_dtype(df[x]):
                raise ValueError(f"X variable '{x}' must be numeric/ordinal")

    def _normalize_data(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Normalize variables to [0, 1] scale.

        Args:
            df: DataFrame containing the data
            columns: List of column names to normalize

        Returns:
            DataFrame with normalized columns
        """
        normalized_df = df.copy()
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val == 0:
                # If all values are the same, set to 0.5
                normalized_df[col] = 0.5
            else:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
        return normalized_df

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First point coordinates
            point2: Second point coordinates

        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def generate_tdabm(self, y: str, x_variables: str, radius: float = 0.3, mcp: bool = False) -> str:
        """
        Generate TDABM analysis and store results in corpus metadata.

        Args:
            y: Name of the y variable (continuous)
            x_variables: Comma-separated list of X variable names
            radius: Radius for ball coverage (default 0.3)
            mcp: If True, return metadata as string for LLM interpretation

        Returns:
            Summary message or metadata string if mcp=True
        """
        # Parse X variables
        x_vars = [x.strip() for x in x_variables.split(',')]

        # Validate variables
        self._validate_variables(y, x_vars)

        # Get clean data (drop rows with NaN in relevant columns)
        all_vars = [y] + x_vars
        df_clean = self.corpus.df[all_vars].dropna()

        if len(df_clean) == 0:
            raise ValueError("No valid data points after removing NaN values")

        # Normalize all variables to [0, 1]
        df_normalized = self._normalize_data(df_clean, all_vars)

        # Create point cloud (X variables only)
        point_cloud = df_normalized[x_vars].values
        y_values = df_normalized[y].values

        # Initialize tracking
        covered = set()
        landmarks = []
        landmark_id = 0

        # Get indices for tracking
        indices = list(range(len(point_cloud)))
        random.shuffle(indices)

        # Ball mapper algorithm
        for idx in indices:
            if idx in covered:
                continue

            # Select landmark point
            landmark_point = point_cloud[idx]

            # Find all points within radius
            ball_points = []
            for i in range(len(point_cloud)):
                distance = self._euclidean_distance(point_cloud[i], landmark_point)
                if distance <= radius:
                    ball_points.append(i)
                    covered.add(i)

            # Calculate statistics for this ball
            ball_y_values = y_values[ball_points]
            mean_y = np.mean(ball_y_values)

            # Store landmark information
            landmarks.append({
                'id': f'B{landmark_id}',
                'location': landmark_point.tolist(),
                'point_indices': ball_points,
                'count': len(ball_points),
                'mean_y': float(mean_y)
            })

            landmark_id += 1

        # Find connections between landmarks (non-empty intersections)
        for i, landmark1 in enumerate(landmarks):
            connections = []
            set1 = set(landmark1['point_indices'])

            for j, landmark2 in enumerate(landmarks):
                if i != j:
                    set2 = set(landmark2['point_indices'])
                    if set1.intersection(set2):  # Non-empty intersection
                        connections.append(landmark2['id'])

            landmark1['connections'] = connections

        # Store in corpus metadata
        tdabm_metadata = {
            'y_variable': y,
            'x_variables': x_vars,
            'radius': radius,
            'num_landmarks': len(landmarks),
            'num_points': len(point_cloud),
            'landmarks': landmarks
        }

        self.corpus.metadata['tdabm'] = tdabm_metadata

        # Prepare summary message
        summary = (
            f"TDABM Analysis Complete:\n"
            f"  Y variable: {y}\n"
            f"  X variables: {', '.join(x_vars)}\n"
            f"  Radius: {radius}\n"
            f"  Number of landmark points: {len(landmarks)}\n"
            f"  Total data points: {len(point_cloud)}\n"
            f"  Coverage: All points covered in {len(landmarks)} balls"
        )

        if mcp:
            # Return detailed metadata for LLM interpretation
            import json
            return json.dumps(tdabm_metadata, indent=2)

        return summary
