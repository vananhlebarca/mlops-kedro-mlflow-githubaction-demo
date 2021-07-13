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

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import mlflow
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import azureml.core
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication


def train_model_rf(x: pd.DataFrame, y: pd.DataFrame, parameters: Dict[str, Any]):

    model = RandomForestClassifier(max_depth=parameters['random_forest']['max_depth'],
                                   random_state=parameters['random_forest']['random_state'],
                                   n_estimators=parameters['random_forest']['n_estimators'])

    model.fit(x, y)

    return model


def predict_model_rf(model, x):

    y = model.predict(x)
    return y


def mlflow_metrics_tracking(model, x, y, parameters, mlflow_params):

    # try:
    #     mlflow.create_experiment(name=mlflow_params['experiment_name'])
    #     logging.info("Create A new experiment! Experiment uri: {}.".format(
    #         mlflow.get_tracking_uri()))

    # except Exception as inst:
    #     if 'RESOURCE_ALREADY_EXISTS' in inst.error_code:
    #         logging.info(
    #             'This experiment does exit! Continue set experiment and track!')
    #     else:
    #         logging.info('Experiment can not be created because of ', inst.error_code)

    # mlflow.set_experiment(mlflow_params['experiment_name'])
    experimentid = mlflow.get_experiment_by_name(
        mlflow_params['experiment_name']).experiment_id

    run_name = str(datetime.datetime.now().strftime(
        "%Y%m%d_%H%M_{}".format('leads')))

    with mlflow.start_run(experiment_id=experimentid, run_name=run_name) as run:
    #with mlflow.start_run() as run:

        y_predicted = model.predict(x)
        y_predicted_proba = model.predict_proba(x)[:, 1]
        # log params
        params_dict = {'n_estimators': parameters['random_forest']['n_estimators'],
                       'max_depth': parameters['random_forest']['max_depth'],
                       'random_state': parameters['random_forest']['random_state']}
        mlflow.log_params(params_dict)

        # log metrics
        accuracy = accuracy_score(y, y_predicted)
        auc_score = roc_auc_score(y, y_predicted_proba)

        metrics_dict = {'accuracy': accuracy, 'auc': auc_score}
        mlflow.log_metrics(metrics_dict)

        # log model
        mlflow.sklearn.log_model(model, 'model')

        run_id = run.info.run_id
        print('Run ID: ', run_id)

        model_uri = "runs:/" + run_id + "/model"
        print('model_uri: ', model_uri)

    return {'tracking_uri': mlflow.get_tracking_uri(), 'run_id': run_id, 'model_uri': model_uri}


# def mlflow_metrics_tracking_databricks(model, x, y, parameters, mlflow_params):

#     try:
#         mlflow.create_experiment(name=mlflow_params['experiment_name'])
#         logging.info("Create A new experiment! Experiment path: {}.".format(
#             mlflow_params['tracking_uri']))

#     except Exception as inst:
#         if 'RESOURCE_ALREADY_EXISTS' in inst.error_code:
#             logging.info(
#                 'This experiment does exit! Continue set experiment and track!')
#         else:
#             logging.info('Experiment can not be created because of ', inst.error_code)

#     mlflow.set_experiment(mlflow_params['experiment_name'])
#     experimentid = mlflow.get_experiment_by_name(
#         mlflow_params['experiment_name']).experiment_id

#     run_name = str(datetime.datetime.now().strftime(
#         "%Y%m%d_%H%M_{}".format('leads')))

#     with mlflow.start_run(experiment_id=experimentid, run_name=run_name):

#         y_predicted = model.predict(x)
#         y_predicted_proba = model.predict_proba(x)[:, 1]
#         # log params
#         params_dict = {'n_estimators': parameters['random_forest']['n_estimators'],
#                        'max_depth': parameters['random_forest']['max_depth'],
#                        'random_state': parameters['random_forest']['random_state']}
#         mlflow.log_params(params_dict)

#         # log metrics
#         accuracy = accuracy_score(y, y_predicted)
#         auc_score = roc_auc_score(y, y_predicted_proba)

#         metrics_dict = {'accuracy': accuracy, 'auc': auc_score}
#         mlflow.log_metrics(metrics_dict)

#         # log model
#         mlflow.sklearn.log_model(model, 'model')


# def mlflow_metrics_tracking_local(model, x, y, parameters, mlflow_params):

#     try:
#         mlflow.create_experiment(name=mlflow_params['experiment_name'])
#         logging.info("Create A new experiment! Experiment path: {}.".format(
#             mlflow_params['tracking_uri']))

#     except Exception as inst:
#         if 'RESOURCE_ALREADY_EXISTS' in inst.error_code:
#             logging.info(
#                 'This experiment does exit! Continue set experiment and track!')
#         else:
#             logging.info('Experiment can not be created because of ', inst.error_code)

#     mlflow.set_experiment(mlflow_params['experiment_name'])
#     experimentid = mlflow.get_experiment_by_name(
#         mlflow_params['experiment_name']).experiment_id

#     run_name = str(datetime.datetime.now().strftime(
#         "%Y%m%d_%H%M_{}".format('leads')))

#     with mlflow.start_run(experiment_id=experimentid, run_name=run_name):

#         y_predicted = model.predict(x)
#         y_predicted_proba = model.predict_proba(x)[:, 1]
#         # log params
#         params_dict = {'n_estimators': parameters['random_forest']['n_estimators'],
#                        'max_depth': parameters['random_forest']['max_depth'],
#                        'random_state': parameters['random_forest']['random_state']}
#         mlflow.log_params(params_dict)

#         # log metrics
#         accuracy = accuracy_score(y, y_predicted)
#         auc_score = roc_auc_score(y, y_predicted_proba)

#         metrics_dict = {'accuracy': accuracy, 'auc': auc_score}
#         mlflow.log_metrics(metrics_dict)

#         # log model
#         mlflow.sklearn.log_model(model, 'model')


# def mlflow_metrics_tracking(model, x, y, parameters, mlflow_params):

#     experimentid = mlflow.get_experiment_by_name(
#         mlflow_params['experiment_name']).experiment_id

#     run_name = str(datetime.datetime.now().strftime(
#         "%Y%m%d_%H%M_{}".format('leads')))

#     with mlflow.start_run(experiment_id=experimentid, run_name=run_name) as run:

#         y_predicted = model.predict(x)
#         y_predicted_proba = model.predict_proba(x)[:, 1]
#         # log params
#         params_dict = {'n_estimators': parameters['random_forest']['n_estimators'],
#                        'max_depth': parameters['random_forest']['max_depth'],
#                        'random_state': parameters['random_forest']['random_state']}
#         mlflow.log_params(params_dict)

#         # log metrics
#         accuracy = accuracy_score(y, y_predicted)
#         auc_score = roc_auc_score(y, y_predicted_proba)

#         metrics_dict = {'accuracy': accuracy, 'auc': auc_score}
#         mlflow.log_metrics(metrics_dict)

#         # log model
#         mlflow.sklearn.log_model(model, 'model')

#         run_id = run.info.run_id
#         print('Run ID: ', run_id)

#         model_uri = "runs:/" + run_id + "/model"
#         print('model_uri: ', model_uri)

#     return {'tracking_uri': mlflow.get_tracking_uri(), 'run_id': run_id, 'model_uri': model_uri}
