# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core import expressions

from activitysim.abm.models.util import estimation
from activitysim.abm.models.util import tour_destination
from activitysim.abm.models.location_choice import iterate_location_choice, write_estimation_specs

from activitysim.core.util import assign_in_place


logger = logging.getLogger(__name__)


@inject.step()
def external_school_location(
    persons_merged, persons, households, network_los, chunk_size, trace_hh_id, locutor
):
    """
    External school location choice model

    iterate_location_choice adds location choice column and annotations to persons table
    """

    trace_label = "external_school_location"
    model_settings = config.read_model_settings("external_school_location.yaml")

    estimator = estimation.manager.begin_estimation("external_school_location")
    if estimator:
        write_estimation_specs(
            estimator, model_settings, "external_school_location.yaml"
        )

    persons_df = iterate_location_choice(
        model_settings,
        persons_merged,
        persons,
        households,
        network_los,
        estimator,
        chunk_size,
        trace_hh_id,
        locutor,
        trace_label,
    )

    pipeline.replace_table("persons", persons_df)

    if estimator:
        estimator.end_estimation()


@inject.step()
def external_workplace_location(
    persons_merged, persons, households, network_los, chunk_size, trace_hh_id, locutor
):
    """
    External workplace location choice model

    iterate_location_choice adds location choice column and annotations to persons table
    """

    trace_label = "external_workplace_location"
    model_settings = config.read_model_settings("external_workplace_location.yaml")

    estimator = estimation.manager.begin_estimation("external_workplace_location")
    if estimator:
        write_estimation_specs(
            estimator, model_settings, "external_workplace_location.yaml"
        )

    persons_df = iterate_location_choice(
        model_settings,
        persons_merged,
        persons,
        households,
        network_los,
        estimator,
        chunk_size,
        trace_hh_id,
        locutor,
        trace_label,
    )

    pipeline.replace_table("persons", persons_df)

    if estimator:
        estimator.end_estimation()


@inject.step()
def external_non_mandatory_destination(
    tours, persons_merged, network_los, chunk_size, trace_hh_id
):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    trace_label = "external_non_mandatory_destination"
    model_settings_file_name = "external_non_mandatory_destination.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        config.setting("want_dest_choice_sample_tables")
        and sample_table_name is not None
    )

    tours = tours.to_frame()

    persons_merged = persons_merged.to_frame()

    # choosers are tours - in a sense tours are choosing their destination
    non_mandatory_ext_tours = tours[
        (tours.tour_category == "non_mandatory") & (tours.is_external_tour)
    ]

    # if no external non-mandatory tours
    if non_mandatory_ext_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    estimator = estimation.manager.begin_estimation(
        "external_non_mandatory_destination"
    )
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag="SPEC")
        estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
        estimator.write_table(
            inject.get_injectable("size_terms"), "size_terms", append=False
        )
        estimator.write_table(
            inject.get_table("land_use").to_frame(), "landuse", append=False
        )
        estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        non_mandatory_ext_tours,
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        estimator,
        chunk_size,
        trace_hh_id,
        trace_label,
    )

    if estimator:
        estimator.write_choices(choices_df.choice)
        choices_df.choice = estimator.get_survey_values(
            choices_df.choice, "tours", "destination"
        )
        estimator.write_override_choices(choices_df.choice)
        estimator.end_estimation()

    non_mandatory_ext_tours["destination"] = choices_df.choice

    assign_in_place(tours, non_mandatory_ext_tours[["destination"]])

    if want_logsums:
        non_mandatory_ext_tours[logsum_column_name] = choices_df["logsum"]
        assign_in_place(tours, non_mandatory_ext_tours[[logsum_column_name]])

    pipeline.replace_table("tours", tours)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # save_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
        pipeline.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        tracing.trace_df(
            tours[tours.tour_category == "non_mandatory"],
            label="external_non_mandatory_destination",
            slicer="person_id",
            index_label="tour",
            columns=None,
            warn_if_empty=True,
        )


@inject.step()
def external_joint_tour_destination(
    tours, persons_merged, network_los, chunk_size, trace_hh_id
):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    trace_label = "external_joint_tour_destination"
    model_settings_file_name = "external_joint_tour_destination.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        config.setting("want_dest_choice_sample_tables")
        and sample_table_name is not None
    )

    tours = tours.to_frame()

    persons_merged = persons_merged.to_frame()

    joint_ext_tours = tours[
        (tours.tour_category == "joint")
        & (tours.get("is_external_tour", False) == True)
    ]  # get needed incase no joint tours

    # if no external joint tours
    if joint_ext_tours.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    estimator = estimation.manager.begin_estimation("external_joint_tour_destination")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
        estimator.write_spec(model_settings, tag="SPEC")
        estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
        estimator.write_table(
            inject.get_injectable("size_terms"), "size_terms", append=False
        )
        estimator.write_table(
            inject.get_table("land_use").to_frame(), "landuse", append=False
        )
        estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        joint_ext_tours,
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        estimator,
        chunk_size,
        trace_hh_id,
        trace_label,
    )

    if estimator:
        estimator.write_choices(choices_df.choice)
        choices_df.choice = estimator.get_survey_values(
            choices_df.choice, "tours", "destination"
        )
        estimator.write_override_choices(choices_df.choice)
        estimator.end_estimation()

    joint_ext_tours["destination"] = choices_df.choice

    assign_in_place(tours, joint_ext_tours[["destination"]])

    if want_logsums:
        joint_ext_tours[logsum_column_name] = choices_df["logsum"]
        assign_in_place(tours, joint_ext_tours[[logsum_column_name]])

    pipeline.replace_table("tours", tours)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # save_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
        pipeline.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        tracing.trace_df(
            tours[tours.tour_category == "non_mandatory"],
            label="external_joint_tour_destination",
            slicer="person_id",
            index_label="tour",
            columns=None,
            warn_if_empty=True,
        )
