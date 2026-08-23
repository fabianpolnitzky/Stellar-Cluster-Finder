def convert_to_galactic(
    df,
    ra_col="ra",
    dec_col="dec",
    parallax_col="parallax",
    pm_ra_col="pm_ra",
    pm_dec_col="pm_dec",
    radial_velocity_col="radial_velocity",
):
    """
    Converts right ascension, declination, proper motion, parallax, and distance of stars to galactic coordinates
    using astropy.

    Parameters:
    df (pandas.DataFrame): Input dataframe containing the star data.
    ra_col (str): Name of the column containing right ascension values. Defaults to "ra".
    dec_col (str): Name of the column containing declination values. Defaults to "dec".
    parallax_col (str): Name of the column containing parallax values. Defaults to "parallax".
    pm_ra_col (str): Name of the column containing proper motion in right ascension values. Defaults to "pm_ra".
    pm_dec_col (str): Name of the column containing proper motion in declination values. Defaults to "pm_dec".
    radial_velocity_col (str): Name of the column containing radial velocity values. Defaults to "radial_velocity".

    Returns:
    pandas.DataFrame: Output dataframe containing the converted galactic coordinates.
    """
    import astropy.units as u
    import pandas as pd
    from astropy.coordinates import Galactic, SkyCoord

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Select the columns containing right ascension, declination, proper motion, parallax, and radial velocity values
    ra = df[ra_col].values * u.degree
    dec = df[dec_col].values * u.degree
    parallax = df[parallax_col].values
    pm_ra = df[pm_ra_col].values * u.mas / u.year
    pm_dec = df[pm_dec_col].values * u.mas / u.year
    radial_velocity = df[radial_velocity_col].values * u.km / u.s

    # Create SkyCoord object with the input values
    distance = (1000.0 / parallax) * u.pc
    skycoord = SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        frame="icrs",
        pm_ra_cosdec=pm_ra,
        pm_dec=pm_dec,
        radial_velocity=radial_velocity,
    )
    # Convert to galactic coordinates
    galactic_coord = skycoord.transform_to(Galactic)

    # Calculate the three dimensional coordinates X, Y, and Z
    x = galactic_coord.cartesian.x.to(u.pc).value
    y = galactic_coord.cartesian.y.to(u.pc).value
    z = galactic_coord.cartesian.z.to(u.pc).value
    # Add the three dimensional coordinates to the output dataframe
    df["X"] = x
    df["Y"] = y
    df["Z"] = z

    # Calculate the velocities U, V, and W using astropy methods
    galactic_vel = galactic_coord.velocity.d_xyz.to(u.km / u.s)
    u = galactic_vel[0].value
    v = galactic_vel[1].value
    w = galactic_vel[2].value
    # Add the velocities to the output dataframe
    df["U"] = u
    df["V"] = v
    df["W"] = w
    return df


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Generate random values for right ascension, declination, parallax, proper motion in RA, proper motion in Dec, and radial velocity
    n = 10
    data = {
        "ra": np.random.uniform(0, 360, n),
        "dec": np.random.uniform(-90, 90, n),
        "parallax": np.random.uniform(1, 100, n),
        "pm_ra": np.random.uniform(-10, 10, n),
        "pm_dec": np.random.uniform(-10, 10, n),
        "radial_velocity": np.random.uniform(-200, 200, n),
    }
    df = pd.DataFrame(data)

    # Use the function to convert to galactic coordinates
    df_galactic = convert_to_galactic(df)

    # Print the output dataframe
    print(df_galactic)
