import pandas as pd


def open_fits(filename):
    """
    Opens a FITS file and converts it to a pandas array.

    Args:
        filename (str): The path to the FITS file.

    Returns:
        pandas.DataFrame: A DataFrame containing the FITS file data.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    import astropy.io.fits as fits

    with fits.open(filename) as hdul:
        data = hdul[0].data

    # Ensure the data is in native byte order
    if hasattr(data, "dtype") and data.dtype.byteorder != "=":
        data = data.astype(data.dtype.newbyteorder("="))

    return pd.DataFrame(data)


def save_dataframe_to_parquet(dataframe, filename):
    """
    Saves a pandas dataframe to a parquet file.

    Args:
        dataframe (pandas.DataFrame): The dataframe to save.
        filename (str): The path to the parquet file.

    Returns:
        None
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    dataframe.to_parquet(filename)


def open_parquet_to_dataframe(filename):
    """
    Opens a parquet file and converts it to a pandas dataframe.

    Args:
        filename (str): The path to the parquet file.

    Returns:
        pandas.DataFrame: A DataFrame containing the parquet file data.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    return pd.read_parquet(filename)


if __name__ == "__main__":
    # Create a small FITS file
    import astropy.io.fits as fits

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    hdu = fits.PrimaryHDU(data)
    fits.writeto("small.fits", hdu.data, hdu.header, overwrite=True)

    # Open the FITS file and convert it to a pandas DataFrame
    df = open_fits("small.fits")
    print(df)

    # Save the pandas DataFrame to a parquet file
    save_dataframe_to_parquet(df, "small.parquet")

    # Open the parquet file and convert it to a pandas DataFrame
    df2 = open_parquet_to_dataframe("small.parquet")
    print(df2)
