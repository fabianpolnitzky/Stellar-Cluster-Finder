import pandas as pd


def load_fits(filename):
    """Open a FITS table file (such as a Gaia catalogue export) and return it as a DataFrame.

    Gaia FITS files store their catalogue as a table in an extension HDU, not in the primary
    HDU, so this reads the first table HDU found in the file. Astropy also takes care of the
    byte order and of masked/missing values during the conversion.

    Args:
        filename (str): Path to the FITS file.

    Returns:
        pandas.DataFrame: The FITS table as a DataFrame.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    from astropy.table import Table

    return Table.read(filename).to_pandas()


def save_parquet(dataframe, filename):
    """Save a DataFrame to a Parquet file.

    Args:
        dataframe (pandas.DataFrame): The DataFrame to save.
        filename (str): Path to the Parquet file to write.

    Returns:
        None
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    dataframe.to_parquet(filename)


def load_parquet(filename):
    """Open a Parquet file and return it as a DataFrame.

    Args:
        filename (str): Path to the Parquet file.

    Returns:
        pandas.DataFrame: The Parquet file as a DataFrame.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    return pd.read_parquet(filename)


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    from astropy.table import Table

    # Anchor the output at the project's examples folder, so the test writes to the same place
    # no matter which directory it is started from.
    examples = Path(__file__).resolve().parents[2] / "examples"
    examples.mkdir(exist_ok=True)
    fits_path = str(examples / "small.fits")
    parquet_path = str(examples / "small.parquet")

    # Build a small Gaia-like table; the column names match convert_to_galactic's defaults.
    rng = np.random.default_rng(42)
    n = 20
    table = Table(
        {
            "ra": rng.uniform(0, 360, n),
            "dec": rng.uniform(-90, 90, n),
            "parallax": rng.uniform(1, 20, n),
            "pmra": rng.uniform(-10, 10, n),
            "pmdec": rng.uniform(-10, 10, n),
            "radial_velocity": rng.uniform(-50, 50, n),
        }
    )
    table.write(fits_path, overwrite=True)

    # Round-trip FITS -> DataFrame -> Parquet -> DataFrame and check nothing is lost.
    from_fits = load_fits(fits_path)
    assert list(from_fits.columns) == list(table.colnames), "FITS columns changed on load"
    assert len(from_fits) == n, "FITS row count changed on load"

    save_parquet(from_fits, parquet_path)
    from_parquet = load_parquet(parquet_path)
    assert from_parquet.equals(from_fits), "Parquet round-trip changed the data"

    print(f"files.py: round-tripped {n} rows through FITS and Parquet into examples/")
    print(from_parquet.head())
