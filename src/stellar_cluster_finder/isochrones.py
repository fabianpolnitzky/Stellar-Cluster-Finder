import numpy as np
import pandas as pd

# Evolutionary stages in a PARSEC table, in the order a star passes through them. From 8 on the
# models describe thermally pulsing AGB stars and remnants, which are drawn at absurd colours
# and magnitudes and are not what a star cluster is dated by.
FIRST_LATE_STAGE = 8


def load_isochrones(filename):
    """Read a PARSEC/CMD isochrone table and return it as a DataFrame.

    Reads the plain text file that the CMD web form (http://stev.oapd.inaf.it/cmd) produces.
    The column names are taken from the commented header line the file carries, so whichever
    quantities were asked for come through under their own names. Two columns are added for
    convenience: ``age_myr``, the age in millions of years instead of its logarithm, and
    ``bp_rp``, the colour G_BP - G_RP, under the same name the Gaia catalogue uses for it, so
    that stars and models can be plotted against the same column name.

    The file has to come from the Gaia photometric system, otherwise it carries no Gaia
    magnitudes to compare Gaia stars with.

    Args:
        filename (str): Path to the isochrone file downloaded from the CMD form.

    Returns:
        pandas.DataFrame: One row per model star, with all columns of the file plus age_myr
            and bp_rp.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    with open(filename, encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    # The CMD file describes itself: the last commented line before the numbers lists the
    # column names, so the table can be read whatever was ticked on the form.
    header = None
    for line in text.splitlines():
        if line.startswith("#") and "Zini" in line and "logAge" in line:
            header = line.lstrip("# ").split()
            break
    if header is None:
        raise ValueError(
            f"'{filename}' has no CMD column header line (a comment line listing 'Zini' and "
            f"'logAge'), so it does not look like an isochrone table from the CMD form."
        )

    isochrones = pd.read_csv(filename, sep=r"\s+", comment="#", names=header)

    missing = [col for col in ("G_BPmag", "G_RPmag", "Gmag") if col not in isochrones.columns]
    if missing:
        raise ValueError(
            f"'{filename}' has no Gaia magnitudes ({missing} are missing). Download the "
            f"isochrones again and set the photometric system on the CMD form to the Gaia one."
        )

    isochrones["age_myr"] = (10.0 ** isochrones["logAge"] / 1.0e6).round(3)
    isochrones["bp_rp"] = isochrones["G_BPmag"] - isochrones["G_RPmag"]

    ages = np.sort(isochrones["age_myr"].unique())
    print(
        f"load_isochrones: {len(isochrones)} model stars, {len(ages)} ages from "
        f"{ages[0]:.0f} to {ages[-1]:.0f} million years."
    )
    return isochrones


def get_isochrone(isochrones, age_myr, include_late_stages=False):
    """Pick the isochrone of one age out of the table, ready to be drawn as a line.

    The file holds many ages at once. This picks the one closest to the age you ask for and
    hands back its model stars sorted by mass, which is the order they lie along the curve in
    a colour-magnitude diagram -- drawn in any other order the line zigzags across the plot.
    Which age was actually available is printed, since the file only contains the steps that
    were asked for on the CMD form.

    The very last evolutionary stages are left out by default. They are thermally pulsing giant
    stars and stellar remnants, a handful of model points that sit at colours and magnitudes far
    outside any real observation and would stretch the plot until the rest of the curve is a
    flat line at the bottom.

    Args:
        isochrones (pandas.DataFrame): Table as returned by load_isochrones.
        age_myr (float): Wanted age in millions of years.
        include_late_stages (bool): Whether to keep thermally pulsing giants and remnants.
            Defaults to False.

    Returns:
        pandas.DataFrame: The rows of the one isochrone, sorted by initial mass.
    """
    if not isinstance(isochrones, pd.DataFrame):
        raise TypeError("isochrones must be a pandas DataFrame")
    if not isinstance(age_myr, (int, float)):
        raise TypeError("age_myr must be a number")
    if not isinstance(include_late_stages, bool):
        raise TypeError("include_late_stages must be True or False")
    if age_myr <= 0:
        raise ValueError("age_myr must be larger than zero")
    missing = [col for col in ("age_myr", "Mini") if col not in isochrones.columns]
    if missing:
        raise ValueError(
            f"These columns are not in the table: {missing}. Read the file with load_isochrones "
            f"so that the ages are prepared."
        )

    ages = np.sort(isochrones["age_myr"].unique())
    closest = float(ages[np.abs(ages - age_myr).argmin()])
    if abs(closest - age_myr) > 1e-3 * max(age_myr, 1.0):
        print(f"get_isochrone: {age_myr:g} million years is not in the file; using {closest:g} instead.")

    one = isochrones[np.isclose(isochrones["age_myr"], closest)]
    if not include_late_stages and "label" in one.columns:
        one = one[one["label"] < FIRST_LATE_STAGE]
    return one.sort_values("Mini")


def add_absolute_magnitude(
    df,
    magnitude_col="phot_g_mean_mag",
    distance_column="distance_pc",
    absolute_column="abs_g_mag",
):
    """Turn the measured brightness of each star into the brightness it would have at 10 parsec.

    What Gaia measures is how bright a star appears, which depends as much on its distance as
    on the star itself. An isochrone says how bright the model stars really are, so the two can
    only be compared once the distance is divided out. That is the absolute magnitude,
    M = m - 5 * log10(distance in pc) + 5.

    You need the distance for this, so run add_distance or convert_to_galactic first. Stars
    without a usable distance or without a measured brightness get NaN and are reported.

    Args:
        df (pandas.DataFrame): Input DataFrame with the star data.
        magnitude_col (str): Name of the measured magnitude column. Defaults to
            "phot_g_mean_mag".
        distance_column (str): Name of the distance column in parsec. Defaults to "distance_pc".
        absolute_column (str): Name of the absolute magnitude column to add. Defaults to
            "abs_g_mag".

    Returns:
        pandas.DataFrame: A copy of the input with the absolute magnitude column added.
    """
    import warnings

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    for name, value in (
        ("magnitude_col", magnitude_col),
        ("distance_column", distance_column),
        ("absolute_column", absolute_column),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
    missing = [col for col in (magnitude_col, distance_column) if col not in df.columns]
    if missing:
        raise ValueError(
            f"These columns are not in the DataFrame: {missing}. Available columns: {list(df.columns)}. "
            f"The distance comes from add_distance or convert_to_galactic; the magnitude has to be "
            f"part of your Gaia download."
        )

    df = df.copy()

    magnitude = pd.to_numeric(df[magnitude_col], errors="coerce").to_numpy(dtype=float)
    distance = pd.to_numeric(df[distance_column], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(magnitude) & np.isfinite(distance) & (distance > 0)

    absolute = np.full(len(df), float("nan"))
    absolute[usable] = magnitude[usable] - 5.0 * np.log10(distance[usable]) + 5.0
    df[absolute_column] = absolute

    n_unusable = int((~usable).sum())
    if n_unusable:
        warnings.warn(
            f"{n_unusable} of {len(df)} rows have no usable '{magnitude_col}' or "
            f"'{distance_column}'; their '{absolute_column}' is set to NaN.",
            stacklevel=2,
        )
    return df


if __name__ == "__main__":
    import warnings
    from pathlib import Path

    examples = Path(__file__).resolve().parents[2] / "examples"
    examples.mkdir(exist_ok=True)
    iso_path = str(examples / "small_isochrones.dat")

    # A miniature CMD file: the same comment header the real one carries, two ages, and a few
    # model stars each, including one late-stage point that has to be dropped by default.
    columns = "Zini MH logAge Mini label Gmag G_BPmag G_RPmag"
    rows = []
    for log_age, offset in ((8.0, 0.0), (8.17609, 0.5)):
        for mass, label in ((0.5, 1), (1.0, 1), (2.0, 1), (5.0, 7), (5.1, 9)):
            bright = 10.0 - 2.0 * mass + offset
            rows.append(f"0.0152 0.015 {log_age:.5f} {mass:.4f} {label} {bright:.3f} {bright + 0.6:.3f} {bright:.3f}")
    Path(iso_path).write_text("# a small test file\n# " + columns + "\n" + "\n".join(rows) + "\n")

    isochrones = load_isochrones(iso_path)
    assert len(isochrones) == 10, "not every model star was read"
    assert set(columns.split()).issubset(isochrones.columns), "the header line was not used for the column names"
    assert {"age_myr", "bp_rp"}.issubset(isochrones.columns), "age_myr and bp_rp were not added"
    assert abs(isochrones["age_myr"].min() - 100.0) < 1e-6, "logAge 8 is not 100 million years"
    assert abs(isochrones["age_myr"].max() - 150.0) < 0.01, "logAge 8.17609 is not 150 million years"

    # An age from the file comes back whole, minus the late stages, and sorted along the curve.
    one = get_isochrone(isochrones, 100.0)
    assert len(one) == 4, "the late evolutionary stage was not dropped"
    assert (one["label"] < FIRST_LATE_STAGE).all(), "a remnant survived the default filter"
    assert list(one["Mini"]) == sorted(one["Mini"]), "the isochrone is not sorted by mass"
    assert len(get_isochrone(isochrones, 100.0, include_late_stages=True)) == 5, (
        "include_late_stages did not keep the remnant"
    )

    # An age that is not in the file falls back to whichever of the two steps is nearer.
    assert abs(get_isochrone(isochrones, 140.0)["age_myr"].iloc[0] - 150.0) < 0.01, (
        "an age between the steps did not fall back to the closest one"
    )
    assert abs(get_isochrone(isochrones, 110.0)["age_myr"].iloc[0] - 100.0) < 0.01, (
        "an age between the steps did not fall back to the closest one"
    )

    # Absolute magnitude: a star at 10 pc keeps its brightness, one at 100 pc is 5 mag brighter.
    stars = pd.DataFrame(
        {
            "phot_g_mean_mag": [10.0, 10.0, 10.0, np.nan],
            "distance_pc": [10.0, 100.0, np.nan, 50.0],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_absolute = add_absolute_magnitude(stars)
    assert any("abs_g_mag" in str(entry.message) for entry in caught), "unusable rows did not warn"
    assert len(with_absolute) == 4, "add_absolute_magnitude changed the row count"
    assert abs(with_absolute.loc[0, "abs_g_mag"] - 10.0) < 1e-9, "a star at 10 pc must keep its magnitude"
    assert abs(with_absolute.loc[1, "abs_g_mag"] - 5.0) < 1e-9, "a star at 100 pc must be 5 magnitudes brighter"
    assert with_absolute.loc[[2, 3], "abs_g_mag"].isna().all(), "rows without distance or magnitude must be NaN"
    assert "abs_g_mag" not in stars.columns, "add_absolute_magnitude must not modify its input"

    print(f"isochrones.py: read {len(isochrones)} model stars at {isochrones['age_myr'].nunique()} ages")
    print("isochrones.py: get_isochrone drops late stages, sorts by mass and falls back to the closest age")
    print("isochrones.py: absolute magnitudes follow M = m - 5 log10(d) + 5 and mark unusable rows NaN")
