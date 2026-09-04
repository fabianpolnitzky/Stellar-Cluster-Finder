def add_distance(df, parallax_col="parallax", distance_column="distance_pc"):
    """Turn Gaia parallaxes into distances in parsec.

    A star's parallax is the small yearly wobble it appears to make because the Earth moves
    around the Sun; the nearer the star, the larger the wobble. The distance follows from it
    as 1000 / parallax, with the parallax in milliarcseconds and the distance in parsec.

    This is a plain division, not a change of coordinate system, so you can use it on the Gaia
    table as it comes and pick your cluster by distance before converting anything. Stars whose
    parallax is missing or not positive get NaN, because no distance can be derived from such a
    measurement; the number of those rows is reported as a warning.

    Args:
        df (pandas.DataFrame): Input DataFrame with the star data.
        parallax_col (str): Name of the parallax column, in milliarcseconds. Defaults to "parallax".
        distance_column (str): Name of the distance column to add, in parsec. Defaults to "distance_pc".

    Returns:
        pandas.DataFrame: A copy of the input with the distance column added.
    """
    import warnings

    import numpy as np
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not isinstance(parallax_col, str):
        raise TypeError("parallax_col must be a string")
    if not isinstance(distance_column, str):
        raise TypeError("distance_column must be a string")
    if parallax_col not in df.columns:
        raise ValueError(
            f"Column '{parallax_col}' is not in the DataFrame. Available columns: {list(df.columns)}. "
            f"If your data uses a different name, pass it as an argument, for example "
            f"add_distance(df, parallax_col='plx')."
        )

    df = df.copy()

    # Only a positive parallax gives a distance. Real Gaia catalogues contain some zero,
    # negative and missing parallaxes; those rows get NaN instead of a nonsensical distance.
    parallax = pd.to_numeric(df[parallax_col], errors="coerce").to_numpy(dtype=float)
    usable = parallax > 0
    distance = np.full(len(df), float("nan"))
    distance[usable] = 1000.0 / parallax[usable]
    df[distance_column] = distance

    n_unusable = int((~usable).sum())
    if n_unusable:
        warnings.warn(
            f"{n_unusable} of {len(df)} rows have a missing or non-positive '{parallax_col}'; "
            f"their '{distance_column}' is set to NaN.",
            stacklevel=2,
        )
    return df


def convert_to_galactic(
    df,
    ra_col="ra",
    dec_col="dec",
    parallax_col="parallax",
    pmra_col="pmra",
    pmdec_col="pmdec",
    radial_velocity_col="radial_velocity",
):
    """Convert Gaia astrometry to Galactic Cartesian positions and space velocities.

    Takes right ascension, declination, parallax, proper motion and radial velocity and adds
    the distance in parsec, the Galactic Cartesian coordinates X, Y, Z (in parsec) and the
    velocities U, V, W (in km/s) using astropy. The default column names match the Gaia
    catalogue.

    This is the step from what Gaia measured to where the stars actually are. You do not need
    it to pick a cluster out of the raw catalogue -- distance, sky position and proper motion
    are enough for that -- but you do need it to see your selection in space, and to hand
    positions and velocities to a clustering algorithm.

    The returned DataFrame always has the same number of rows as the input. Rows whose
    parallax is missing or not positive get NaN in all seven new columns and trigger a warning,
    because the distance (1000 / parallax) cannot be derived from such a parallax.

    Gaia measures a radial velocity for far fewer stars than it measures a parallax for, and
    without one there is no full space velocity. Those rows keep their position -- distance_pc,
    X, Y and Z are computed as usual -- but get NaN in U, V and W, and are reported in a second
    warning. This is worth knowing before clustering: a run on X, Y, Z uses the whole table,
    while a run on U, V, W or on all six columns silently drops every star without a radial
    velocity.

    Args:
        df (pandas.DataFrame): Input DataFrame with the star data.
        ra_col (str): Name of the right ascension column, in degrees. Defaults to "ra".
        dec_col (str): Name of the declination column, in degrees. Defaults to "dec".
        parallax_col (str): Name of the parallax column, in milliarcseconds. Defaults to "parallax".
        pmra_col (str): Name of the proper-motion-in-RA column, in mas/yr. Defaults to "pmra".
        pmdec_col (str): Name of the proper-motion-in-Dec column, in mas/yr. Defaults to "pmdec".
        radial_velocity_col (str): Name of the radial velocity column, in km/s. Defaults to "radial_velocity".

    Returns:
        pandas.DataFrame: A copy of the input with the columns distance_pc (pc), X, Y, Z (pc)
            and U, V, W (km/s) added.
    """
    import warnings

    import astropy.units as u
    import pandas as pd
    from astropy.coordinates import Galactic, SkyCoord

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    needed = [ra_col, dec_col, parallax_col, pmra_col, pmdec_col, radial_velocity_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(
            f"These columns are not in the DataFrame: {missing}. Available columns: {list(df.columns)}. "
            f"If your data uses different names, pass them as arguments, for example "
            f"convert_to_galactic(df, pmra_col='pm_ra')."
        )

    df = df.copy()
    new_cols = ["distance_pc", "X", "Y", "Z", "U", "V", "W"]
    for col in new_cols:
        df[col] = float("nan")

    # The distance is computed as distance_pc = 1000 / parallax_mas, which is only meaningful
    # for a positive parallax. Real Gaia catalogues contain some zero, negative or missing
    # parallaxes; those rows keep NaN in the new columns and are skipped in the conversion.
    valid = df[parallax_col] > 0
    n_invalid = int((~valid).sum())
    if n_invalid:
        warnings.warn(
            f"{n_invalid} of {len(df)} rows have a missing or non-positive '{parallax_col}'; "
            f"their distance, galactic coordinates and velocities are set to NaN.",
            stacklevel=2,
        )
    if not valid.any():
        return df

    # A star without a radial velocity still has a position; only its space velocity is
    # unknown. astropy fills U, V and W with NaN for those rows on its own, but silently, and
    # the loss only shows up much later as stars missing from a velocity clustering run.
    no_velocity = valid & pd.to_numeric(df[radial_velocity_col], errors="coerce").isna()
    n_no_velocity = int(no_velocity.sum())
    if n_no_velocity:
        warnings.warn(
            f"{n_no_velocity} of {int(valid.sum())} rows with a usable parallax have no "
            f"'{radial_velocity_col}'; their 'U', 'V' and 'W' are set to NaN, while "
            f"'distance_pc', 'X', 'Y' and 'Z' are computed as usual.",
            stacklevel=2,
        )

    sub = df.loc[valid]

    # Read the input columns and attach the physical units astropy expects.
    ra = sub[ra_col].values * u.degree
    dec = sub[dec_col].values * u.degree
    parallax = sub[parallax_col].values
    pmra = sub[pmra_col].values * u.mas / u.year
    pmdec = sub[pmdec_col].values * u.mas / u.year
    radial_velocity = sub[radial_velocity_col].values * u.km / u.s

    # Build the SkyCoord and transform it from ICRS to the Galactic frame.
    distance = (1000.0 / parallax) * u.pc
    skycoord = SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        frame="icrs",
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=radial_velocity,
    )
    galactic_coord = skycoord.transform_to(Galactic)

    # Cartesian positions X, Y, Z in parsec.
    x = galactic_coord.cartesian.x.to(u.pc).value
    y = galactic_coord.cartesian.y.to(u.pc).value
    z = galactic_coord.cartesian.z.to(u.pc).value

    # Cartesian velocities U, V, W in km/s.
    galactic_vel = galactic_coord.velocity.d_xyz.to(u.km / u.s)
    u_vel = galactic_vel[0].value
    v_vel = galactic_vel[1].value
    w_vel = galactic_vel[2].value

    # Fill the results back into the rows that had a usable parallax; the rest stay NaN.
    df.loc[valid, "distance_pc"] = distance.value
    df.loc[valid, "X"] = x
    df.loc[valid, "Y"] = y
    df.loc[valid, "Z"] = z
    df.loc[valid, "U"] = u_vel
    df.loc[valid, "V"] = v_vel
    df.loc[valid, "W"] = w_vel
    return df


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 10
    df = pd.DataFrame(
        {
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-90, 90, n),
            "parallax": rng.uniform(1, 100, n),
            "pmra": rng.uniform(-10, 10, n),
            "pmdec": rng.uniform(-10, 10, n),
            "radial_velocity": rng.uniform(-200, 200, n),
        }
    )
    # Mimic real Gaia data by making two parallaxes unusable (one negative, one missing) and
    # by leaving one star without a radial velocity, which is the common case in Gaia.
    df.loc[0, "parallax"] = -1.0
    df.loc[1, "parallax"] = np.nan
    df.loc[2, "radial_velocity"] = np.nan

    # add_distance works on the raw catalogue, without any coordinate transformation.
    with_distance = add_distance(df)
    assert len(with_distance) == n, "add_distance changed the row count"
    assert with_distance.loc[[0, 1], "distance_pc"].isna().all(), "unusable parallaxes should give NaN distance"
    assert with_distance.loc[2:, "distance_pc"].notna().all(), "usable parallaxes should give a distance"
    expected = 1000.0 / df.loc[2, "parallax"]
    assert abs(with_distance.loc[2, "distance_pc"] - expected) < 1e-9, "distance is not 1000 / parallax"
    assert "X" not in with_distance.columns, "add_distance must not do the galactic conversion"

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = convert_to_galactic(df)
    messages = [str(entry.message) for entry in caught]
    assert any("parallax" in text for text in messages), "unusable parallaxes did not warn"
    assert any("radial_velocity" in text for text in messages), "missing radial velocity did not warn"
    new_cols = ["distance_pc", "X", "Y", "Z", "U", "V", "W"]
    position_cols = ["distance_pc", "X", "Y", "Z"]
    velocity_cols = ["U", "V", "W"]

    # Both functions have to agree on the distance, or a selection made before the conversion
    # would no longer match the data after it.
    same = result["distance_pc"].notna()
    assert (result.loc[same, "distance_pc"] - with_distance.loc[same, "distance_pc"]).abs().max() < 1e-9, (
        "add_distance and convert_to_galactic disagree about the distance"
    )

    # Same number of rows, new columns present, bad-parallax rows are NaN and the rest are not.
    assert len(result) == n, "row count changed"
    assert set(new_cols).issubset(result.columns), "galactic columns missing"
    assert result.loc[[0, 1], new_cols].isna().all().all(), "bad-parallax rows should be NaN"
    assert result.loc[3:, new_cols].notna().all().all(), "good rows should not be NaN"

    # A star without a radial velocity keeps its position and loses only its space velocity,
    # which is what makes a clustering run on X, Y, Z bigger than one on U, V, W.
    assert result.loc[2, position_cols].notna().all(), "a missing radial velocity must not lose the position"
    assert result.loc[2, velocity_cols].isna().all(), "a missing radial velocity must give NaN in U, V, W"

    print(f"coordinates.py: {n} rows in, {n} rows out, 2 rows set to NaN for unusable parallax")
    print("coordinates.py: a star without a radial velocity keeps X, Y, Z and loses only U, V, W")
    print("coordinates.py: add_distance agrees with convert_to_galactic and adds no galactic columns")
    print(result)
