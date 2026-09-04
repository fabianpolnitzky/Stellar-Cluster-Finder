import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Filled shapes only, so they stay visible with the marker edge switched off in crowded plots.
# Groups are told apart by shape as well as by colour: a plot that leans on colour alone stops
# working in greyscale and for a reader with a colour vision deficiency.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

# The label find_one_cluster gives to stars in no cluster; drawn apart from the numbered groups.
_NOISE_MARKER = "."

# A histogram has bars rather than points, so the same job falls to hatching: the stacked
# sections stay tellable apart once the colours are gone. The first is left plain, which is
# itself a pattern, so a two-group split reads as plain against hatched.
_HATCHES = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo", "**", "--", "||"]


def _solid_legend(ax):
    """Draw the legend's symbols full size and opaque.

    The points themselves are shrunk and faded to open up a crowded field, but the legend is the
    key to which shape means which group, so it has to stay readable: at alpha 0.25 and a
    four-pixel marker its symbols are all but invisible.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    for handle in legend.legend_handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1)
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([60])
        elif hasattr(handle, "set_markersize"):
            handle.set_markersize(8)


def _check_limits(limits, name):
    """Reject anything that is not a (low, high) pair of numbers.

    Args:
        limits (tuple or None): The value to check.
        name (str): The argument name, so the message says which one was wrong.
    """
    if limits is None:
        return
    if not isinstance(limits, (tuple, list)) or len(limits) != 2:
        raise TypeError(f"{name} must be a tuple of two numbers, for example (0, 500), or None.")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in limits):
        raise TypeError(f"both values in {name} must be numbers.")
    if limits[0] >= limits[1]:
        raise ValueError(f"the first value of {name} ({limits[0]}) must be smaller than the second ({limits[1]}).")


def _point_style(df, alpha):
    """Work out how faint and how small the points must be for a frame of this size.

    Transparency on its own does not open up a crowded field: at the default marker size
    thousands of points tile the panel whatever the alpha, because each one is large and carries
    a pale edge that reads as a tile border. The marker therefore shrinks with the crowd and
    loses that edge as well. All three follow the row count, so passing an alpha by hand still
    gets the small edgeless marker.

    Args:
        df (pandas.DataFrame): The frame being plotted; only its length is used.
        alpha (float or None): Opacity chosen by the caller, or None to measure one.

    Returns:
        tuple: The alpha, the marker size and the marker edge width to draw with.
    """
    if alpha is not None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number between 0 and 1, or None")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be greater than 0 and at most 1, got {alpha}")

    crowded = len(df) > 1000
    if alpha is None:
        alpha = max(0.05, (1000 / len(df)) ** 0.5) if crowded else 1.0
        if alpha < 1:
            print(f"plot: {len(df)} points drawn with alpha={alpha:.2f}; pass alpha=... to change it.")

    size = max(3.0, 36 * (1000 / len(df)) ** 0.75) if crowded else 36
    return alpha, size, 0 if crowded else 0.5


def plot_histogram(
    dataframe,
    column,
    color_column=None,
    bins="auto",
    title="Histogram",
    xlabel="Value",
    ylabel="Counts",
    vline=None,
    xlim=None,
    save_path=None,
):
    """Plot a histogram of one column of a DataFrame with seaborn.

    Stars without a usable measurement in the plotted column are left out of the histogram and
    counted in a short message. Gaia does not measure every star equally well, and derived
    quantities such as the distance are NaN wherever the parallax could not be used; those
    stars would otherwise vanish silently and make the totals hard to follow.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        column (str): Name of the column to plot.
        color_column (str, optional): Column to split and colour the bars by. The bars are
            stacked, so each one keeps its full height and shows the split as coloured
            sections. Defaults to None.
        bins (str or int, optional): Number of bins, or a binning rule name. Defaults to "auto".
        title (str, optional): Plot title. Defaults to "Histogram".
        xlabel (str, optional): Label for the x-axis. Defaults to "Value".
        ylabel (str, optional): Label for the y-axis. Defaults to "Counts".
        vline (float, optional): x position of a dashed vertical reference line. Defaults to None.
        xlim (tuple, optional): (left, right) section of the x-axis to show. The bins are placed
            inside this section, so it zooms into a peak instead of just cropping the picture.
            Defaults to None, which shows everything.
        save_path (str, optional): If given, the figure is written to this path. Defaults to None.

    The figure is deliberately not returned. A notebook draws every figure a cell creates,
    and it draws a returned figure a second time as the cell's result, so returning it made
    a plain call show the same plot twice. Use plt.gcf() to get hold of the figure if you
    want to keep tweaking it.
    """
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
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' is not in the DataFrame. Available columns: {list(dataframe.columns)}")
    _check_limits(xlim, "xlim")

    # Leave out the stars Gaia could not measure well enough. They are NaN (or infinite, where
    # a division by a parallax of almost zero blew up) and carry no information for the shape
    # of the distribution, but they do belong to the table, so say how many were dropped.
    data = dataframe
    if pd.api.types.is_numeric_dtype(dataframe[column]):
        import numpy as np

        usable = np.isfinite(dataframe[column].to_numpy(dtype=float))
        n_unusable = int((~usable).sum())
        if n_unusable:
            print(
                f"plot_histogram: {n_unusable} of {len(dataframe)} stars have no usable '{column}' and are not shown."
            )
        data = dataframe.loc[usable]

    fig = plt.figure()

    # binrange makes the bins follow the zoom, so a narrow peak keeps its detail. multiple
    # ="stack" splits each bar between the colours instead of drawing them on top of each other,
    # which is what you want when the colour says "selected or not": the bars keep their full
    # height and the selection is a coloured slice of them, not a bite taken out.
    palette = None
    if color_column is not None and data[color_column].nunique() <= 20:
        palette = sns.color_palette("colorblind", data[color_column].nunique())
    sns.histplot(data=data, x=column, hue=color_column, bins=bins, binrange=xlim, multiple="stack", palette=palette)

    # Give each stacked section its own hatch, so the split survives greyscale and colour vision
    # deficiency exactly as the marker shapes do on the scatter plots. Handles are matched to
    # containers by colour rather than by position, because seaborn stacks them in reverse order
    # and a mismatch would label the wrong pattern.
    if color_column is not None:
        from matplotlib.patches import Patch

        legend = plt.gca().get_legend()
        raw_handles = legend.legend_handles if legend is not None else []
        handles = [handle for handle in raw_handles if isinstance(handle, Patch)]
        for index, container in enumerate(reversed(plt.gca().containers)):
            hatch = _HATCHES[index % len(_HATCHES)]
            for patch in container:
                patch.set_hatch(hatch)
                patch.set_edgecolor("white")
                patch.set_linewidth(0.5)
            for handle in handles:
                if handle.get_facecolor() == container[0].get_facecolor():
                    handle.set_hatch(hatch)
                    handle.set_edgecolor("white")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if vline is not None:
        plt.axvline(x=vline, color="r", linestyle="--")

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    if save_path is not None:
        fig.savefig(save_path)


def plot_dataframe(
    df, x_col, y_col, title, xlabel, ylabel, color_col=None, marker_col=None, alpha=None, xlim=None, ylim=None
):
    """Draw a scatter plot of two DataFrame columns and return the figure.

    This is the low-level builder used by plot_and_save; it does not call plt.show or save.

    Args:
        df (pandas.DataFrame): The DataFrame to plot.
        x_col (str): Column to use for the x-axis.
        y_col (str): Column to use for the y-axis.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color_col (str, optional): Column to colour the points by. Defaults to None.
        marker_col (str, optional): Column to choose point markers by. Rarely needed: left at
            None, a color_col holding a handful of groups supplies the shapes itself, so the
            groups differ in shape as well as in colour without being asked to. Name a column
            here only to take the shapes from something other than the colour. Defaults to None.
        alpha (float, optional): How opaque each point is, from just above 0 to 1. A dense field
            hides its own structure: with thousands of solid points a cluster is only more black
            among black, while faded points add up where the stars crowd together and the
            overdensity becomes visible. Left at None it is measured from the number of points,
            because no fixed value suits both a small test frame and a Gaia catalogue; frames of
            up to 1000 points stay fully opaque. The chosen value is printed. Defaults to None.
        xlim (tuple, optional): (left, right) section of the x-axis to show. Points outside it
            stay in the table and are simply not drawn. Defaults to None, which shows everything.
        ylim (tuple, optional): (bottom, top) section of the y-axis to show. Zooming both axes at
            once is how you get at a clump that sits inside a wider spread, such as the cluster
            in the proper motion, where a handful of fast outliers otherwise stretch the axes so
            far that everything interesting is squeezed into a corner. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure that was created.
    """
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
    _check_limits(xlim, "xlim")
    _check_limits(ylim, "ylim")

    # Measure the crowding from the points that will actually be on show. Zooming into a small
    # patch is the main reason to pass limits at all, and a patch holding 300 stars should be
    # drawn like a 300-star frame rather than faded as though all 15000 were still in the way.
    visible = df
    if xlim is not None:
        visible = visible[(visible[x_col] >= xlim[0]) & (visible[x_col] <= xlim[1])]
    if ylim is not None:
        visible = visible[(visible[y_col] >= ylim[0]) & (visible[y_col] <= ylim[1])]

    alpha, size, edge_width = _point_style(visible, alpha)

    # Colour alone is not enough to tell groups apart, so a colour column holding a handful of
    # groups gives the marker shapes as well unless the caller named another column for them.
    # A column with many distinct values is a continuous quantity, which has no groups to shape.
    if color_col is not None and marker_col is None and df[color_col].nunique() <= len(_MARKERS):
        marker_col = color_col

    if marker_col is not None and df[marker_col].nunique() > len(_MARKERS):
        raise ValueError(
            f"'{marker_col}' has {df[marker_col].nunique()} different values, more than the "
            f"{len(_MARKERS)} marker shapes available. Pick a column with fewer groups for marker_col."
        )

    fig = plt.figure()
    options = {"x": x_col, "y": y_col, "data": df, "alpha": alpha, "s": size, "linewidth": edge_width}
    if color_col is not None:
        options["hue"] = color_col
        # seaborn only takes a list of colours for a categorical hue; a continuous one gets a
        # colour map of its own, which we leave alone.
        if df[color_col].nunique() <= 20:
            options["palette"] = sns.color_palette("colorblind", df[color_col].nunique())
    if marker_col is not None:
        options["style"] = marker_col
        options["markers"] = _MARKERS[: df[marker_col].nunique()]

    sns.scatterplot(**options)
    _solid_legend(plt.gca())

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    return fig


def plot_ellipse(params):
    """Add an unfilled ellipse to the current axes.

    Args:
        params (dict): Must contain the keys:
            x0 (float): x-coordinate of the ellipse centre.
            y0 (float): y-coordinate of the ellipse centre.
            width (float): Full width of the ellipse.
            height (float): Full height of the ellipse.
            angle (float): Rotation of the ellipse in degrees.

    Returns:
        matplotlib.patches.Ellipse: The ellipse artist that was added to the current axes.
    """
    from matplotlib.patches import Ellipse

    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    if not all(key in params for key in ["x0", "y0", "width", "height", "angle"]):
        raise ValueError("params must contain the keys 'x0', 'y0', 'width', 'height', and 'angle'")
    if not all(isinstance(value, (int, float)) for value in params.values()):
        raise TypeError("all values in params must be numbers")

    ellipse = Ellipse(
        (params["x0"], params["y0"]),
        params["width"],
        params["height"],
        angle=params["angle"],
        fill=False,
    )
    plt.gca().add_artist(ellipse)
    return ellipse


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
    line_x=None,
    line_y=None,
    line_label=None,
    invert_yaxis=False,
    alpha=None,
    xlim=None,
    ylim=None,
):
    """Draw a scatter plot with optional ellipse and reference line, and optionally save it.

    Args:
        df (pandas.DataFrame): The DataFrame to plot.
        x_col (str): Column to use for the x-axis.
        y_col (str): Column to use for the y-axis.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str, optional): If given, the figure is written to this path. Defaults to None.
        color_col (str, optional): Column to colour the points by. Defaults to None.
        marker_col (str, optional): Column to choose point markers by. Rarely needed: left at
            None, a color_col holding a handful of groups supplies the shapes itself, so the
            groups differ in shape as well as in colour without being asked to. Name a column
            here only to take the shapes from something other than the colour. Defaults to None.
        ellipse_params (dict, optional): Ellipse to overlay; see plot_ellipse. Defaults to None.
        line_x (array-like, optional): x-values of an extra dashed reference line. Defaults to None.
        line_y (array-like, optional): y-values of an extra dashed reference line. Defaults to None.
        line_label (str, optional): Name of that line in the legend, for example the age of an
            isochrone. Without it the line is drawn but not named. Defaults to None.
        invert_yaxis (bool): Whether to let the y-axis run downwards. Needed for a
            colour-magnitude diagram, where a smaller magnitude means a brighter star and the
            bright stars belong at the top. Defaults to False.
        alpha (float, optional): How opaque each point is, from just above 0 to 1. Fading the
            points is what makes a cluster visible in a crowded field: solid points cover each
            other, faded ones add up where the stars are dense. Left at None it is measured from
            the number of points and printed. Defaults to None.
        xlim (tuple, optional): (left, right) section of the x-axis to show. Defaults to None.
        ylim (tuple, optional): (bottom, top) section of the y-axis to show. Use the two together
            to zoom in on a clump inside a wider spread — the proper motion is the case this
            exists for, where a few fast stars stretch both axes until the cluster is a dot in
            the middle. The transparency is then measured from the stars left inside the window,
            so a close zoom is not faded as heavily as the whole field. Defaults to None.

    The figure is deliberately not returned. A notebook draws every figure a cell creates,
    and it draws a returned figure a second time as the cell's result, so returning it made
    a plain call show the same plot twice. Use plt.gcf() to get hold of the figure if you
    want to keep tweaking it.
    """
    fig = plot_dataframe(df, x_col, y_col, title, xlabel, ylabel, color_col, marker_col, alpha, xlim, ylim)

    if line_x is not None and line_y is not None:
        plt.plot(line_x, line_y, color="red", linestyle="--", label=line_label)
        if line_label is not None:
            plt.legend()

    if ellipse_params is not None:
        plot_ellipse(ellipse_params)

    # A magnitude counts backwards: the smaller the number, the brighter the star. Without this
    # a colour-magnitude diagram comes out upside down.
    if invert_yaxis:
        plt.gca().invert_yaxis()

    if save_path is not None:
        fig.savefig(save_path)


def plot_3d(df, x_col, y_col, z_col, title, xlabel, ylabel, zlabel, save_path=None, color_col=None, alpha=None):
    """Draw a 3D scatter plot of three DataFrame columns, and optionally save it.

    This is the plot for looking at the stars in space and checking whether a cluster really
    holds together in three dimensions. To turn and zoom it with the mouse in a Jupyter
    notebook, run the command ``%matplotlib widget`` in a cell before calling this
    function; without it the notebook draws a still image you cannot rotate.

    Pass the label column produced by find_one_cluster as ``color_col`` to see each cluster in
    its own colour *and its own marker shape*, so the groups stay apart in greyscale and for a
    reader with a colour vision deficiency. Columns with up to 20 distinct values are treated
    as categories and get a legend, with the noise label ``-1`` in grey and its own shape;
    columns with more distinct values are
    treated as a continuous quantity and get a colour bar.

    Args:
        df (pandas.DataFrame): The DataFrame to plot.
        x_col (str): Column to use for the x-axis.
        y_col (str): Column to use for the y-axis.
        z_col (str): Column to use for the z-axis.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis.
        save_path (str, optional): If given, the figure is written to this path. Defaults to None.
        color_col (str, optional): Column to colour the points by, such as a cluster label
            column. Defaults to None.
        alpha (float, optional): How opaque each point is, from just above 0 to 1. Depth makes a
            3D scatter even more crowded than a flat one, because points behind each other land
            on top of each other; fading them lets the dense places show through. Left at None it
            is measured from the number of points and printed, exactly as for the 2D plots, and
            the same value is used for every group so the colours stay comparable. Defaults to
            None.

    The figure is deliberately not returned. A notebook draws every figure a cell creates,
    and it draws a returned figure a second time as the cell's result, so returning it made
    a plain call show the same plot twice. Use plt.gcf() to get hold of the figure if you
    want to keep tweaking it.
    """
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
    if color_col is not None and not isinstance(color_col, str):
        raise TypeError("color_col must be a string or None")

    alpha, size, edge_width = _point_style(df, alpha)
    style = {"alpha": alpha, "s": size, "linewidths": edge_width}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if color_col is None:
        ax.scatter(df[x_col], df[y_col], df[z_col], **style)
    elif df[color_col].nunique() <= 20:
        # Few distinct values, so treat them as categories: one colour, one marker shape and one
        # legend entry each. Cluster labels from find_one_cluster land here. Shape carries the
        # same distinction as colour, so the groups survive greyscale and colour vision
        # deficiency; the noise group keeps its own grey and its own shape on top of that.
        values = sorted(df[color_col].dropna().unique())
        clusters = [value for value in values if value != -1]
        if len(clusters) > len(_MARKERS):
            raise ValueError(
                f"'{color_col}' has {len(clusters)} groups, more than the {len(_MARKERS)} marker "
                "shapes available. Cluster into fewer groups to keep them tellable apart."
            )
        colors = sns.color_palette("colorblind", max(len(clusters), 1))

        for value in values:
            group = df[df[color_col] == value]
            if value == -1:
                # The label find_one_cluster gives to stars that belong to no cluster.
                ax.scatter(
                    group[x_col],
                    group[y_col],
                    group[z_col],
                    color="lightgrey",
                    marker=_NOISE_MARKER,
                    label="noise (-1)",
                    **style,
                )
            else:
                index = clusters.index(value)
                ax.scatter(
                    group[x_col],
                    group[y_col],
                    group[z_col],
                    color=colors[index],
                    marker=_MARKERS[index],
                    label=str(value),
                    **style,
                )
        ax.legend(title=color_col)
        _solid_legend(ax)
    else:
        # Many distinct values, so treat the column as a continuous quantity.
        points = ax.scatter(df[x_col], df[y_col], df[z_col], c=df[color_col], **style)
        fig.colorbar(points, ax=ax, label=color_col)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)

    if save_path is not None:
        fig.savefig(save_path)


if __name__ == "__main__":
    from pathlib import Path

    from matplotlib.figure import Figure

    # Anchor the output at the project's examples folder, so the test writes to the same place
    # no matter which directory it is started from.
    examples = Path(__file__).resolve().parents[2] / "examples"
    examples.mkdir(exist_ok=True)

    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 25, 30]})
    scatter_path = str(examples / "example_plot.png")
    plot_and_save(
        df,
        "x",
        "y",
        "Example Plot",
        "X-axis",
        "Y-axis",
        save_path=scatter_path,
        color_col="x",
        ellipse_params={"x0": 2.5, "y0": 22.5, "width": 4, "height": 8, "angle": 45},
    )
    fig = plt.gcf()
    assert isinstance(fig, Figure) and Path(scatter_path).exists(), "plot_and_save did not produce a file"

    df["z"] = df["x"] * 2
    scatter3d_path = str(examples / "example_3d.png")
    plot_3d(df, "x", "y", "z", "Example Plot", "X-axis", "Y-axis", "Z-axis", save_path=scatter3d_path)
    fig = plt.gcf()
    assert isinstance(fig, Figure) and Path(scatter3d_path).exists(), "plot_3d did not produce a file"

    # Colour the same points by a cluster label column, the way the students see their clusters.
    # Label -1 is the noise label find_one_cluster uses for stars in no cluster.
    df["cluster_labels_HDBSCAN"] = [-1, 0, 0, 1]
    clusters3d_path = str(examples / "example_3d_clusters.png")
    plot_3d(
        df,
        "x",
        "y",
        "z",
        "Clusters in 3D",
        "X (pc)",
        "Y (pc)",
        "Z (pc)",
        save_path=clusters3d_path,
        color_col="cluster_labels_HDBSCAN",
    )
    fig = plt.gcf()
    legend_labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert legend_labels == ["noise (-1)", "0", "1"], f"unexpected legend entries: {legend_labels}"
    assert Path(clusters3d_path).exists(), "plot_3d did not produce a file for the coloured plot"

    ages = pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "Gender": ["Male", "Female"] * 5,
        }
    )
    hist_path = str(examples / "example_histogram.png")
    plot_histogram(
        ages,
        column="Age",
        color_column="Gender",
        bins=10,
        title="Age Distribution by Gender",
        xlabel="Age",
        ylabel="Counts",
        vline=50,
        save_path=hist_path,
    )
    fig = plt.gcf()
    assert isinstance(fig, Figure) and Path(hist_path).exists(), "plot_histogram did not produce a file"

    # The coloured bars have to be stacked, not drawn on top of each other. Stacking gives the
    # upper group a non-zero baseline; layering would leave every bar sitting at y = 0 and make
    # a selection look like a bite out of the bar instead of a coloured part of it.
    assert any(patch.get_y() > 0 for patch in fig.axes[0].patches), "the coloured bars are not stacked"

    # Stars without a usable measurement must not reach the histogram. NaN turns up wherever
    # Gaia's parallax could not be used, and it has to be visibly dropped rather than silently.
    gap = pd.DataFrame({"distance_pc": [100.0, 110.0, 120.0, float("nan"), float("inf"), 130.0]})
    gaps_path = str(examples / "example_histogram_gaps.png")
    plot_histogram(
        gap,
        column="distance_pc",
        bins=4,
        title="Distances, gaps left out",
        xlabel="distance (pc)",
        save_path=gaps_path,
    )
    fig = plt.gcf()
    plotted = sum(patch.get_height() for patch in fig.axes[0].patches)
    assert plotted == 4, f"expected the 4 usable stars in the histogram, got {plotted}"
    assert Path(gaps_path).exists(), "plot_histogram did not produce a file"

    # xlim zooms into a section of the axis and places the bins inside it.
    plot_histogram(gap, column="distance_pc", bins=4, title="Zoom", xlabel="distance (pc)", xlim=(100, 120))
    fig = plt.gcf()
    assert fig.axes[0].get_xlim() == (100.0, 120.0), "xlim was not applied to the axis"
    zoomed = sum(patch.get_height() for patch in fig.axes[0].patches)
    assert zoomed == 3, f"expected the 3 stars between 100 and 120 pc, got {zoomed}"

    # A histogram has bars rather than points, so its split is carried by hatching instead.
    split = pd.DataFrame({"v": list(range(60)), "g": ["a", "b", "c"] * 20})
    plot_histogram(split, column="v", color_column="g", bins=10, title="Split", xlabel="v")
    hatches = {container[0].get_hatch() for container in plt.gca().containers}
    assert len(hatches) == 3, f"expected one hatch pattern per group, got {hatches}"

    # Groups must never be told apart by colour alone: the plot has to survive greyscale and a
    # reader with a colour vision deficiency, so every group gets its own shape as well.
    labelled = pd.DataFrame({"x": range(60), "y": range(60), "cluster": [-1, 0, 1, 2] * 15})

    plot_and_save(labelled, "x", "y", "Groups in 2D", "x (pc)", "y (pc)", color_col="cluster")
    shapes = {str(path.vertices) for path in plt.gca().collections[0].get_paths()}
    assert len(shapes) == 4, f"expected one marker shape per group, got {len(shapes)}"

    plot_3d(labelled, "x", "y", "x", "Groups in 3D", "x (pc)", "y (pc)", "x (pc)", color_col="cluster")
    shapes = [str(c.get_paths()[0].vertices) for c in plt.gca().collections]
    assert len(shapes) == 4 and len(set(shapes)) == 4, f"3D groups share a marker shape: {len(set(shapes))}"

    # A column with more groups than there are shapes has to fail loudly rather than silently
    # reusing one, which would make two different clusters look like the same thing.
    crowd = pd.DataFrame({"x": range(24), "y": range(24), "cluster": list(range(12)) * 2})
    for call in (
        lambda: plot_and_save(crowd, "x", "y", "Too many", "x (pc)", "y (pc)", marker_col="cluster"),
        lambda: plot_3d(crowd, "x", "y", "x", "Too many", "x (pc)", "y (pc)", "x (pc)", color_col="cluster"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("more groups than marker shapes should have been rejected")

    # Both axes can be zoomed, which is how a clump inside a wider spread is found. The limits
    # have to reach the axes, and the transparency has to follow the stars left inside them.
    wide = pd.DataFrame({"x": list(range(3000)) + [50000], "y": list(range(3000)) + [50000]})
    plot_and_save(wide, "x", "y", "Zoomed", "x (pc)", "y (pc)", xlim=(0, 100), ylim=(0, 100))
    assert plt.gca().get_xlim() == (0.0, 100.0), "xlim was not applied to the 2D axis"
    assert plt.gca().get_ylim() == (0.0, 100.0), "ylim was not applied to the 2D axis"
    assert plt.gca().collections[0].get_alpha() == 1.0, "a zoom holding 101 points should not be faded"

    plot_and_save(wide, "x", "y", "Not zoomed", "x (pc)", "y (pc)")
    unzoomed = plt.gca().collections[0].get_alpha()
    assert unzoomed is not None and unzoomed < 1, "the unzoomed frame should still be faded"

    for bad, err in (((5,), TypeError), (("a", 2), TypeError), ((5, 5), ValueError), ((9, 2), ValueError)):
        try:
            plot_and_save(wide, "x", "y", "Bad", "x (pc)", "y (pc)", ylim=bad)
        except err:
            pass
        else:
            raise AssertionError(f"ylim={bad!r} should have been rejected with {err.__name__}")

    # A 3D scatter is crowded in the same way, and gets the same treatment through the same
    # helper — including in the categorical branch, where every group has to share one alpha.
    spread = pd.DataFrame({"x": range(1500), "y": range(1500), "z": range(1500)})
    spread["label"] = [-1, 0, 1] * 500
    plot_3d(spread, "x", "y", "z", "Faded in 3D", "x (pc)", "y (pc)", "z (pc)", color_col="label")
    faded = [c.get_alpha() for c in plt.gca().collections]
    assert faded and all(a is not None and a < 1 for a in faded), f"3D groups were not faded: {faded}"
    assert len(set(faded)) == 1, f"the 3D groups were faded differently: {faded}"

    plot_3d(spread, "x", "y", "z", "Opaque in 3D", "x (pc)", "y (pc)", "z (pc)", alpha=1.0)
    assert plt.gca().collections[0].get_alpha() == 1.0, "alpha=1 did not reach the 3D points"

    try:
        plot_3d(spread, "x", "y", "z", "Bad", "x (pc)", "y (pc)", "z (pc)", alpha=0)
    except ValueError:
        pass
    else:
        raise AssertionError("plot_3d should reject alpha=0")

    # Transparency is what makes a cluster visible in a crowded field, so it has to reach the
    # points and not just the call. A small frame stays opaque; a large one is faded for us.
    dense = pd.DataFrame({"x": list(range(2000)), "y": list(range(2000))})
    plot_dataframe(dense, "x", "y", "Dense", "x", "y")
    measured = plt.gca().collections[0].get_alpha()
    assert measured is not None and measured < 1, f"a dense frame was not faded, got {measured}"

    plot_dataframe(df, "x", "y", "Sparse", "x", "y")
    assert plt.gca().collections[0].get_alpha() == 1.0, "a small frame should stay opaque"

    plot_and_save(dense, "x", "y", "Chosen by hand", "x", "y", alpha=0.25)
    assert plt.gca().collections[0].get_alpha() == 0.25, "alpha passed by hand did not reach the points"

    for bad, err in ((0, ValueError), (1.5, ValueError), ("0.5", TypeError), (True, TypeError)):
        try:
            plot_dataframe(df, "x", "y", "Bad alpha", "x", "y", alpha=bad)
        except err:
            pass
        else:
            raise AssertionError(f"alpha={bad!r} should have been rejected with {err.__name__}")

    print("plotting.py: wrote the example_*.png figures to examples/")
    print("plotting.py: histogram leaves out unusable measurements and zooms with xlim")
