# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def lower_columns(data, categorical_columns):
    """ import dataset and lower columns name
    Args:
        data: pd.Dataframe
    Returns:
        data_lower: pd.Dataframe
    """

    data.columns = map(str.lower, data.columns)
    data[categorical_columns] = data[categorical_columns].apply(lambda x: x.str.lower())
    return data


def split_data(data, leads_response_column,  split_ratio) -> Dict[str, Any]:
    # def split_data(data, parameters) -> Dict[str, Any]:
    """split data for training, remove extras
    Args:
        data: pd.DataFrame
        leads_response_column: str
        example_test_data_ratio: float
    Returns:
    """

    leads_x = data.drop(leads_response_column, axis=1)
    leads_y = data[leads_response_column]

    #split_ratio = parameters['split_ratio']
    train_x, test_x, train_y, test_y = train_test_split(leads_x,
                                                        leads_y,
                                                        train_size=split_ratio,
                                                        test_size=(1-split_ratio),
                                                        random_state=5050)

    return dict(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
    )


def scaler_fit(data, numeric_columns):
    """Fit Standard Scaler with numeric columns of train_x
    Args:
        data: pd.DataFrame
        numeric_columns: List 
    Returns:
        scaler: sklearn.preprocessing.StandardScaler
    """

    scaler = StandardScaler()
    scaler = scaler.fit(data[numeric_columns])

    return scaler


def onehotencoder_fit(data, categorical_columns):

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoder.fit(data[categorical_columns])

    return encoder


def pre_process_leads_data(fitted_onehotencoder,
                           fitted_scaler,
                           data,
                           numeric_columns,
                           categorical_columns,
                           train_df_columns=None):

    # create new df with selected columns

    _data = data[set(numeric_columns + categorical_columns)].copy()

    # scale the numeric columns with the pre-built scaler
    _data[numeric_columns] = fitted_scaler.transform(_data[numeric_columns])

    _data_encoded = fitted_onehotencoder.transform(_data[categorical_columns])
    _data_encoded_df = pd.DataFrame(_data_encoded)

    _data.drop(categorical_columns, axis=1, inplace=True)

    _data = pd.concat([_data, _data_encoded_df], axis=1)

    if train_df_columns:
        _data = _data.reindex(columns=train_df_columns, fill_value=0)

    return _data
