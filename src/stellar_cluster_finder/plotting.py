import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_histogram(dataframe, column, color_column=None, bins="auto", title="Histogram", xlabel="Value", ylabel="Counts", vline=None, save_path=None):
    """
    Plots a histogram of a specified column in a dataframe using seaborn and matplotlib.

    Args:
        dataframe (pd.DataFrame): The input dataframe.
        column (str): The name of the column to plot.
        bins (str or float, optional): The number of bins to use for the histogram. Defaults to "auto".
        color_column (str, optional): The name of the column to use for coloring the histogram. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Histogram".
        xlabel (str, optional): The label of the x-axis. Defaults to "Value".
        ylabel (str, optional): The label of the y-axis. Defaults to "Frequency".
        vline (float, optional): The position of the vertical line. Defaults to None.
        save_path (str, optional): The path to save the plot. Defaults to None.

    Returns:
        None
    """
    # Check input types
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if not isinstance(column, str):
        raise TypeError("column must be a string.")
    if color_column is not None and not isinstance(color_column, str):
        raise TypeError("color_column must be a string or None.")
    if not isinstance(title, str):
        raise TypeError("title must be a string.")
    if not isinstance(xlabel, str):
        raise TypeError("xlabel must be a string.")
    if not isinstance(ylabel, str):
        raise TypeError("ylabel must be a string.")
    if vline is not None and not isinstance(vline, (int, float)):
        raise ValueError("vline must be a number or None.")

    fig = plt.figure()

    # Plot histogram
    sns.histplot(data=dataframe, x=column, hue=color_column, bins=bins)

    # Set title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add vertical line
    if vline is not None:
        plt.axvline(x=vline, color="r", linestyle="--")

    # Show plot
    plt.show()

    if save_path is not None:
        # Save the plot to a file
        fig.savefig(save_path)

    # Close the plot
    plt.close()

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
    save_path=None,
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
        save_path (str, optional): The path to save the plot. Defaults to None.
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

    plt.show()

    if save_path is not None:
        # Save the plot to a file
        fig.savefig(save_path)

    # Close the plot
    plt.close(fig)

def plot_3d(df, x_col, y_col, z_col, title, xlabel, ylabel, zlabel, save_path=None):
    """
    Plots three columns from a dataframe in 3d.

    Parameters:
        df (pandas.DataFrame): The dataframe to plot.
        x_col (str): The column to use for the x-axis.
        y_col (str): The column to use for the y-axis.
        z_col (str): The column to use for the z-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        zlabel (str): The label for the z-axis.
        save_path (str, optional): The path to save the plot. Defaults to None.

    Returns:
        None
    """
    # Check for input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(x_col, str):
        raise TypeError("x_col must be a string")
    if not isinstance(y_col, str):
        raise TypeError("y_col must be a string")
    if not isinstance(z_col, str):
        raise TypeError("z_col must be a string")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(xlabel, str):
        raise TypeError("xlabel must be a string")
    if not isinstance(ylabel, str):
        raise TypeError("ylabel must be a string")
    if not isinstance(zlabel, str):
        raise TypeError("zlabel must be a string")
    if save_path is not None and not isinstance(save_path, str):
        raise TypeError("save_path must be a string or None")

    # Create a figure object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Plot the data
    ax.scatter(df[x_col], df[y_col], df[z_col])

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)

    plt.show()

    if save_path is not None:
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

    df["z"] = df["x"] * 2
    plot_3d(df, "x", "y", "z", "Example Plot", "X-axis", "Y-axis", "Z-axis", "example_3d.png")

    # Generate test data
    data = {
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"]
    }
    df = pd.DataFrame(data)

    # Plot histogram
    plot_histogram(df, column="Age", color_column="Gender", bins=10, title="Age Distribution by Gender", xlabel="Age", ylabel="Counts", vline=50, save_path="example_histogram.png")