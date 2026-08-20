import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_dataframe(
    df, x_col, y_col, title, xlabel, ylabel, color_col=None, marker_col=None
):
    """
    Plots two columns from a dataframe against each other and returns a figure object.

    Args:
        df (pandas.DataFrame): The dataframe to plot.
        x_col (str): The column to use for the x-axis.
        y_col (str): The column to use for the y-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        color_col (str, optional): The column to use for coloring the points. Defaults to None.
        marker_col (str, optional): The column to use for marking the points. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure object created for the plot.
    """
    # Check for input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(x_col, str):
        raise TypeError("x_col must be a string")
    if not isinstance(y_col, str):
        raise TypeError("y_col must be a string")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(xlabel, str):
        raise TypeError("xlabel must be a string")
    if not isinstance(ylabel, str):
        raise TypeError("ylabel must be a string")
    if color_col is not None and not isinstance(color_col, str):
        raise TypeError("color_col must be a string or None")
    if marker_col is not None and not isinstance(marker_col, str):
        raise TypeError("marker_col must be a string or None")

    # Create a figure object
    fig = plt.figure()
    # Plot the data
    if color_col is None and marker_col is None:
        sns.scatterplot(x=x_col, y=y_col, data=df)
    elif color_col is not None and marker_col is None:
        sns.scatterplot(x=x_col, y=y_col, hue=color_col, data=df)
    elif color_col is None and marker_col is not None:
        sns.scatterplot(x=x_col, y=y_col, style=marker_col, data=df)
    else:
        sns.scatterplot(x=x_col, y=y_col, hue=color_col, style=marker_col, data=df)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return fig


def plot_ellipse(params):
    """
    Plots an ellipse based on the given parameters.

    Args:
        params (dict): A dictionary with the following keys:
            - x0 (float): The x-coordinate of the center of the ellipse.
            - y0 (float): The y-coordinate of the center of the ellipse.
            - width (float): The width of the ellipse.
            - height (float): The height of the ellipse.
            - angle (float): The angle of rotation of the ellipse in degrees.

    Returns:
        None
    """
    from matplotlib.patches import Ellipse

    # Check for input types
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    if not all(key in params for key in ["x0", "y0", "width", "height", "angle"]):
        raise ValueError(
            "params must contain the keys 'x0', 'y0', 'width', 'height', and 'angle'"
        )
    if not all(isinstance(value, (int, float)) for value in params.values()):
        raise TypeError("all values in params must be numbers")

    # Plot the ellipse
    ellipse = Ellipse(
        (params["x0"], params["y0"]),
        params["width"],
        params["height"],
        angle=params["angle"],
        fill=False,
    )
    plt.gca().add_artist(ellipse)


def plot_and_save(
    df,
    x_col,
    y_col,
    title,
    xlabel,
    ylabel,
    save_path,
    color_col=None,
    marker_col=None,
    ellipse_params=None,
):
    """
    Plots two columns from a dataframe against each other and saves the plot to a file.

    Parameters:
        df (pandas.DataFrame): The dataframe to plot.
        x_col (str): The column to use for the x-axis.
        y_col (str): The column to use for the y-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        save_path (str): The path to save the plot.
        color_col (str, optional): The column to use for coloring the points. Defaults to None.
        marker_col (str, optional): The column to use for marking the points. Defaults to None.
        ellipse_params (dict, optional): The parameters of the ellipse to plot. Defaults to None.

    Returns:
        None
    """
    # Create a figure object and plot the data
    fig = plot_dataframe(df, x_col, y_col, title, xlabel, ylabel, color_col, marker_col)
    # Plot the ellipse if provided
    if ellipse_params is not None:
        plot_ellipse(ellipse_params)

    # Save the plot to a file
    fig.savefig(save_path)

    # Close the plot
    plt.close(fig)


if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 25, 30]})
    plot_and_save(
        df,
        "x",
        "y",
        "Example Plot",
        "X-axis",
        "Y-axis",
        "example_plot.png",
        color_col="x",
        ellipse_params={"x0": 2.5, "y0": 22.5, "width": 4, "height": 8, "angle": 45},
    )
