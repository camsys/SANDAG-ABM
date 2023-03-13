# ActivitySim
# See full license in LICENSE.txt.
import datetime
import logging
import sys

import numpy as np
import pandas as pd

from activitysim.core import config, inject, pipeline
from activitysim.core.config import setting
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import ContainerClient
from io import StringIO

logger = logging.getLogger(__name__)


@inject.step()
def write_to_datalake(output_dir):
    """
    Write pipeline tables as csv files (in output directory) as specified by output_tables list
    in settings file.

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

    # Get SAS token from key vault
    # VAULT_URL = os.environ["VAULT_URL"]
    # credential = DefaultAzureCredential()
    # client = KeyClient(vault_url=VAULT_URL, credential=credential)
    # sas_url = client.get_secret(secretName)

    sas_url = ""
    container = ContainerClient.from_container_url(sas_url)
    datetimestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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

            # write to data lake
            output = StringIO()
            output = df.to_csv(index_label="idx", encoding="utf-8")
            lake_file = datetimestamp + "/" + file_name
            blob_client = container.upload_blob(name=lake_file, data=output)
