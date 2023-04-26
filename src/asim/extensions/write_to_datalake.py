# ActivitySim
# See full license in LICENSE.txt.
import datetime
import logging
import os
import socket
import uuid

import numpy as np
import pandas as pd

from activitysim.core import config, inject, pipeline
from activitysim.core.config import setting
from azure.storage.blob import ContainerClient
from io import StringIO

logger = logging.getLogger(__name__)


@inject.step()
def write_to_datalake(output_dir):
    """
    Write pipeline tables as csv files to Azure Data Lake Storage as
    specified by output_tables list in settings file.

    Azure CLI must be installed on server.
    Type 'az login' from a windows command prompt.
    This will open a web browser for azure login.

    Environment variables on windows must be set for key vault url
    and secret name on Azure.

    'output_tables' can specify either a list of output tables to include or to skip
    if no output_tables list is specified, then all checkpointed tables will be written

    To write all output tables EXCEPT the households and persons tables:

    ::

      output_tables:
        action: skip
        tables:
          - households
          - persons

    To write ONLY the households table:

    ::

      output_tables:
        action: include
        tables:
           - households

    To write tables into a single HDF5 store instead of individual CSVs, use the h5_store flag:

    ::

      output_tables:
        h5_store: True
        action: include
        tables:
           - households

    Parameters
    ----------
    output_dir: str

    """

    sas_url = os.environ["AZURE_STORAGE_SAS_TOKEN"]

    # create Azure ContainerClient
    container = ContainerClient.from_container_url(sas_url)

    # generate the timestamp as a datetime object
    now = datetime.datetime.now()

    # format the timestamp as a string in a consistent format
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # create folder structure for hierarchical organization of files
    year_folder = now.strftime("%Y")
    month_folder = now.strftime("%m")

    # generate a globally unique identifier
    guid = uuid.uuid4().hex

    output_tables_settings_name = "output_tables"

    output_tables_settings = setting(output_tables_settings_name)

    if output_tables_settings is None:
        logger.info(
            "No output_tables specified in settings file. Nothing to write.")
        return

    action = output_tables_settings.get("action")
    tables = output_tables_settings.get("tables")
    prefix = output_tables_settings.get("prefix", "final_")
    h5_store = output_tables_settings.get("h5_store", False)
    sort = output_tables_settings.get("sort", False)

    registered_tables = pipeline.registered_tables()
    if action == "include":
        # interpret empty or missing tables setting to mean include all registered tables
        output_tables_list = tables if tables is not None else registered_tables
    elif action == "skip":
        output_tables_list = [t for t in registered_tables if t not in tables]
    else:
        raise "expected %s action '%s' to be either 'include' or 'skip'" % (
            output_tables_settings_name,
            action,
        )

    for table_name in output_tables_list:

        if table_name == "checkpoints":
            df = pipeline.get_checkpoints()
        else:
            if table_name not in registered_tables:
                logger.warning("Skipping '%s': Table not found." % table_name)
                continue
            df = pipeline.get_table(table_name)

            if sort:
                traceable_table_indexes = inject.get_injectable(
                    "traceable_table_indexes", {}
                )

                if df.index.name in traceable_table_indexes:
                    df = df.sort_index()
                    logger.debug(
                        f"write_tables sorting {table_name} on index {df.index.name}"
                    )
                else:
                    # find all registered columns we can use to sort this table
                    # (they are ordered appropriately in traceable_table_indexes)
                    sort_columns = [
                        c for c in traceable_table_indexes if c in df.columns
                    ]
                    if len(sort_columns) > 0:
                        df = df.sort_values(by=sort_columns)
                        logger.debug(
                            f"write_tables sorting {table_name} on columns {sort_columns}"
                        )
                    else:
                        logger.debug(
                            f"write_tables sorting {table_name} on unrecognized index {df.index.name}"
                        )
                        df = df.sort_index()

        if h5_store:
            file_path = config.output_file_path("%soutput_tables.h5" % prefix)
            df.to_hdf(file_path, key=table_name, mode="a", format="fixed")
        else:
            file_name = "%s%s.csv" % (prefix, table_name)
            file_path = config.output_file_path(file_name)

            # include the index if it has a name or is a MultiIndex
            write_index = df.index.name is not None or isinstance(
                df.index, pd.MultiIndex
            )

            # extract base filename and extension
            base_filename, ext = os.path.splitext(os.path.basename(file_name))

            # add unique identifier
            df["guid"] = guid

            # add the timestamp as a new column to the DataFrame
            df["timestamp"] = pd.to_datetime(now)

            # Construct the model output filename w guid
            model_output_file = f"{base_filename }_{timestamp_str}_{guid}{ext}"

            # extract table name from base filename, e.g. households, trips, persons, etc.
            tablename = base_filename.split("final_")[1]

            lake_file = (
                f"{tablename}/{year_folder}/{month_folder}/{model_output_file}"
            )

            # write tables to data lake
            output = StringIO()
            output = df.to_csv(
                date_format="%Y-%m-%d %H:%M:%S", index=write_index, encoding="utf-8"
            )
            blob_client = container.upload_blob(
                name=lake_file, data=output, encoding="utf-8"
            )

    # create metadata table
    username = os.getenv('USERNAME')
    machine_name = socket.gethostname()
    commit_hash = os.popen('git rev-parse HEAD').read().strip()
    short_commit_hash = commit_hash[-7:]
    branch_name = os.popen(
        'git rev-parse --abbrev-ref HEAD').read().strip()

    metadata = {'model_run_name': ['abm3_dev_test'],
                'year': ['2022'],
                'user_name': [username],
                'machine_name': [machine_name],
                'commit_hash': [short_commit_hash],
                'branch_name': [branch_name],
                'guid': [guid]
                }

    meta_df = pd.DataFrame(metadata)

    # add the timestamp as a new column to the DataFrame
    meta_df["timestamp"] = pd.to_datetime(now)

    # generate metadata filename and path
    metadata_file = f"scenario_{timestamp_str}_{guid}.csv"
    metadata_path = (
        f"meta_data/{year_folder}/{month_folder}/{metadata_file}"
    )

    # write metadata to data lake
    output = StringIO()
    output = meta_df.to_csv(
        date_format="%Y-%m-%d %H:%M:%S", index=False, encoding="utf-8"
    )

    blob_client = container.upload_blob(
        name=metadata_path, data=output, encoding="utf-8", overwrite=True
    )
