import numpy as np
import pandas as pd


def select_range(df, column, minimum=None, maximum=None, selection_column="selection"):
    """Mark the rows whose value in one column lies between two limits.

    This is the selection that goes with plot_histogram: read the limits of a peak off the
    histogram and keep the stars inside it. Leaving a limit at None means "no limit on that
    side". Missing values are never selected. How many stars ended up selected is printed, so
    you can see straight away whether the limits caught what you meant them to.

    Args:
        df (pandas.DataFrame): Input DataFrame. It is modified in place.
        column (str): Name of the column to apply the limits to.
        minimum (float, optional): Lower limit; rows below it are not selected. Defaults to None.
        maximum (float, optional): Upper limit; rows above it are not selected. Defaults to None.
        selection_column (str): Name of the True/False column to write. Defaults to "selection".

    Returns:
        pandas.DataFrame: The same DataFrame with the True/False selection column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(column, str):
        raise TypeError("column must be a string")
    if minimum is not None and not isinstance(minimum, (int, float)):
        raise TypeError("minimum must be a number or None")
    if maximum is not None and not isinstance(maximum, (int, float)):
        raise TypeError("maximum must be a number or None")
    if not isinstance(selection_column, str):
        raise TypeError("selection_column must be a string")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is not in the DataFrame. Available columns: {list(df.columns)}")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"minimum ({minimum}) is larger than maximum ({maximum}), so nothing could be selected.")

    inside = df[column].notna()
    if minimum is not None:
        inside &= df[column] >= minimum
    if maximum is not None:
        inside &= df[column] <= maximum

    df[selection_column] = inside
    print(f"select_range: {int(inside.sum())} of {len(df)} stars selected into '{selection_column}'.")
    return df


def select_ellipse(df, x_col, y_col, params, selection_column="selection"):
    """Mark the rows that lie inside an ellipse in a two-dimensional plot.

    Use the same ``params`` dictionary you passed to plot_and_save as ``ellipse_params``, so
    that the stars you select are exactly the ones you drew the ellipse around. Stars with a
    missing value in either column are never selected. How many stars ended up inside the
    ellipse is printed, so you can see straight away whether you drew it where you meant to.

    Args:
        df (pandas.DataFrame): Input DataFrame. It is modified in place.
        x_col (str): Column shown on the x-axis of the plot.
        y_col (str): Column shown on the y-axis of the plot.
        params (dict): The ellipse, with the same keys plot_ellipse uses:
            x0 (float): x-coordinate of the ellipse centre.
            y0 (float): y-coordinate of the ellipse centre.
            width (float): Full width of the ellipse before rotation.
            height (float): Full height of the ellipse before rotation.
            angle (float): Rotation of the ellipse in degrees, counterclockwise.
        selection_column (str): Name of the True/False column to write. Defaults to "selection".

    Returns:
        pandas.DataFrame: The same DataFrame with the True/False selection column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(x_col, str):
        raise TypeError("x_col must be a string")
    if not isinstance(y_col, str):
        raise TypeError("y_col must be a string")
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    if not all(key in params for key in ["x0", "y0", "width", "height", "angle"]):
        raise ValueError("params must contain the keys 'x0', 'y0', 'width', 'height', and 'angle'")
    if not all(isinstance(value, (int, float)) for value in params.values()):
        raise TypeError("all values in params must be numbers")
    if params["width"] <= 0 or params["height"] <= 0:
        raise ValueError("width and height of the ellipse must be larger than zero")
    if not isinstance(selection_column, str):
        raise TypeError("selection_column must be a string")
    missing = [col for col in (x_col, y_col) if col not in df.columns]
    if missing:
        raise ValueError(f"These columns are not in the DataFrame: {missing}. Available columns: {list(df.columns)}")

    # Shift the stars so the ellipse sits at the origin, then turn them by minus the ellipse
    # angle. In that turned frame the ellipse lies along the axes and the test is the simple
    # (x/a)^2 + (y/b)^2 <= 1.
    dx = df[x_col].to_numpy(dtype=float) - params["x0"]
    dy = df[y_col].to_numpy(dtype=float) - params["y0"]
    angle = np.radians(params["angle"])
    x_turned = dx * np.cos(angle) + dy * np.sin(angle)
    y_turned = -dx * np.sin(angle) + dy * np.cos(angle)

    a = params["width"] / 2.0
    b = params["height"] / 2.0
    with np.errstate(invalid="ignore"):
        inside = (x_turned / a) ** 2 + (y_turned / b) ** 2 <= 1.0

    df[selection_column] = pd.Series(inside, index=df.index).fillna(False)
    print(f"select_ellipse: {int(df[selection_column].sum())} of {len(df)} stars selected into '{selection_column}'.")
    return df


if __name__ == "__main__":
    # A circle of radius 1 at the origin: only the points closer than 1 belong to it.
    points = pd.DataFrame({"x": [0.0, 0.5, 0.99, 1.01, 5.0, np.nan], "y": [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]})
    circle = {"x0": 0.0, "y0": 0.0, "width": 2.0, "height": 2.0, "angle": 0.0}
    select_ellipse(points, "x", "y", circle, selection_column="in_circle")
    assert list(points["in_circle"]) == [True, True, True, False, False, False], "circle test failed"

    # A long thin ellipse turned by 90 degrees reaches far along y and stays narrow along x.
    turned = pd.DataFrame({"x": [0.0, 0.0, 3.0], "y": [0.0, 3.0, 0.0]})
    select_ellipse(turned, "x", "y", {"x0": 0, "y0": 0, "width": 8, "height": 2, "angle": 90}, selection_column="sel")
    assert list(turned["sel"]) == [True, True, False], "rotated ellipse test failed"

    # The selection has to agree with the ellipse matplotlib actually draws.
    from matplotlib.patches import Ellipse

    rng = np.random.default_rng(0)
    cloud = pd.DataFrame({"x": rng.uniform(-6, 6, 500), "y": rng.uniform(-6, 6, 500)})
    shape = {"x0": 1.0, "y0": -0.5, "width": 5.0, "height": 2.0, "angle": 35.0}
    select_ellipse(cloud, "x", "y", shape, selection_column="sel")
    patch = Ellipse((shape["x0"], shape["y0"]), shape["width"], shape["height"], angle=shape["angle"])
    drawn = patch.get_path().transformed(patch.get_patch_transform()).contains_points(cloud[["x", "y"]].to_numpy())
    assert (cloud["sel"].to_numpy() == drawn).all(), "selection does not match the drawn ellipse"

    # Range selection, including the open-ended and missing-value cases.
    values = pd.DataFrame({"parallax": [1.0, 5.0, 7.0, 9.0, np.nan]})
    select_range(values, "parallax", minimum=5.0, maximum=8.0, selection_column="near")
    assert list(values["near"]) == [False, True, True, False, False], "range test failed"
    select_range(values, "parallax", minimum=5.0, selection_column="open_end")
    assert list(values["open_end"]) == [False, True, True, True, False], "open-ended range test failed"

    # Selections are True/False columns, so they can be combined with & and |.
    combined = values["near"] & values["open_end"]
    assert list(combined) == [False, True, True, False, False], "combining selections failed"

    print(f"selection.py: ellipse selection matches matplotlib for all {len(cloud)} test points")
    print("selection.py: range selection and combining selections behave as documented")
