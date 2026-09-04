# Stellar Cluster Finder

*Deutsche Anleitung für Teilnehmende: [README.de.md](README.de.md)*

This repository is used for the [Physik-Projekt-Tage](https://ppt.physik.rwth-aachen.de/) at
RWTH Aachen to introduce school students to scientific research and programming. The project
is about finding stellar clusters using *Python* and observational data from the *Gaia* space
telescope.

Most stars drift through the Galaxy alone. Some travel in groups: they were born together
from the same cloud of gas and still move together today. This code gives you the tools to go
looking for such a group — first by eye in what the telescope actually measured, then with the
same clustering algorithms used in professional astronomy — and to compare what the different
approaches find.

You are not expected to arrive at one perfect list of cluster members. The point is to try
several ways of looking at the same stars, see what each one is good at, and work out how such
a search is actually done.

> **The data is not included.** You download your own table of stars from the Gaia archive as
> a FITS file and point the code at it. See [Getting the data](#getting-the-data) for the
> columns you need, and [What this has been tested on](#what-this-has-been-tested-on) for the
> clusters it has actually been run against.

---

## For students

### Setup

The project uses [uv](https://docs.astral.sh/uv/) to install everything you need:

```bash
uv sync
uv run jupyter lab
```

That is all. `uv sync` installs Python and every package; `jupyter lab` opens the notebook
environment in your browser.

### Getting the data

Put your Gaia download into the `data/` folder. A plain `SELECT *` works, but it hands you 152
columns to look past and a file roughly ten times larger than it needs to be. These eight are
the ones the code actually uses:

| Column | What it is needed for |
| --- | --- |
| `ra`, `dec` | Position on the sky — selecting by eye, and clustering |
| `parallax` | The distance, via `add_distance` |
| `pmra`, `pmdec` | Proper motion — selecting by eye, and clustering |
| `radial_velocity` | The third velocity component, without which there is no `U, V, W` |
| `phot_g_mean_mag` | Brightness, for the age determination |
| `bp_rp` | Colour, for the age determination |

A cone around a cluster, asking for exactly those, looks like this in the archive's query box:

```sql
SELECT ra, dec, parallax, pmra, pmdec, radial_velocity, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 56.60, 24.11, 4))
  AND parallax BETWEEN 3 AND 12
```

The `CONTAINS(...)` form is what the archive's spatial index is built for; writing the same
cone as `DISTANCE(...) < 4` can be far slower. The parallax limits keep the file small and, as
a side effect, remove every unusable parallax — so nothing in your table will be missing a
distance.

You can add `source_id` if you want an identifier to trace individual stars; nothing in the
code needs it.

### The notebooks

The course itself is three notebooks in `notebooks/`, written in German. Work through them in
order — each hands its result to the next:

| Notebook | What you do |
| --- | --- |
| `01_auswahl_von_hand.ipynb` | Find the cluster yourself in the raw Gaia measurements: select by distance, by sky position and by proper motion, combine the three, then convert to real positions and look at the result in 3D |
| `02_clustering_algorithmen.ipynb` | Let HDBSCAN, DBSCAN and a Gaussian Mixture Model do the same job in position, in velocity and in the full phase space, tune their parameters, and compare with your own selection |
| `03_alter_des_haufens.ipynb` | *Optional.* Work out how old the cluster is by laying model isochrones over its colour-magnitude diagram. Reachable from either of the two notebooks above |

The rest of this section is the same workflow written out as plain code, in case you would
rather build your own notebook or need to look something up.

### The workflow

In a notebook, start by importing the functions you need:

```python
from stellar_cluster_finder import (
    load_fits,
    save_parquet,
    load_parquet,
    add_distance,
    convert_to_galactic,
    plot_histogram,
    plot_and_save,
    plot_3d,
    select_range,
    select_ellipse,
    find_one_cluster,
    add_absolute_magnitude,
    load_isochrones,
    get_isochrone,
)
```

**1. Load your data.** Read the FITS file you downloaded, then save it as a Parquet file —
that format loads much faster, so you only have to read the FITS file once.

```python
stars = load_fits("data/my_gaia_region.fits")
save_parquet(stars, "data/my_gaia_region.parquet")

stars = load_parquet("data/my_gaia_region.parquet")  # from now on, use this
```

What you now have is what Gaia measured: where each star sits on the sky (`ra`, `dec`), its
parallax, how fast it drifts across the sky (`pmra`, `pmdec`) and, for some stars, how fast it
moves towards or away from us (`radial_velocity`).

**2. Turn the parallax into a distance.** The nearer a star, the more it appears to wobble over
a year. That wobble is the parallax, and the distance follows from it directly:

```python
stars = add_distance(stars)  # adds distance_pc
```

You will see a warning that some rows were set to `NaN`. That is normal: the distance is
`1000 / parallax`, which makes no sense when Gaia measured a parallax of zero or less. Those
stars keep their place in the table but get no distance.

**3. Pick a cluster by eye, in the measurements themselves.** Cluster members sit at roughly the
same distance, stand close together on the sky, and drift across it as a group. That is three
different views of the same stars, and you make a selection in each:

```python
plot_histogram(stars, "distance_pc", bins=60, xlabel="distance (pc)", xlim=(0, 500))
select_range(stars, "distance_pc", minimum=130, maximum=160, selection_column="by_distance")

sky = {"x0": 56.5, "y0": 24.0, "width": 4, "height": 4, "angle": 0}
plot_and_save(stars, "ra", "dec", title="On the sky", xlabel="RA (deg)", ylabel="Dec (deg)", ellipse_params=sky)
select_ellipse(stars, "ra", "dec", sky, selection_column="by_sky")

motion = {"x0": 20, "y0": -10, "width": 8, "height": 8, "angle": 0}
plot_and_save(
    stars,
    "pmra",
    "pmdec",
    title="Proper motion",
    xlabel="pmRA (mas/yr)",
    ylabel="pmDec (mas/yr)",
    ellipse_params=motion,
    xlim=(0, 40),  # zoom both axes to get at the clump
    ylim=(-70, -10),
)
select_ellipse(stars, "pmra", "pmdec", motion, selection_column="by_motion")

stars["my_cluster"] = stars["by_distance"] & stars["by_sky"] & stars["by_motion"]
```

Each selection prints how many stars it caught. Draw the plot again with
`color_col="by_motion"` to see what you actually picked, adjust the numbers, and run it again —
that loop is most of the work.

With a whole Gaia field the scatter plots fade their points for you, and print the transparency
they chose. Thousands of solid dots cover one another and turn the cluster into a filled blob;
faded ones pile up where the stars crowd together, so the overdensity shows through. Pass
`alpha=1` to switch the fading off and see the difference, or a smaller value to push it
further.

`xlim` and `ylim` crop the view without touching the table, and the proper motion is where you
will want them. A handful of fast-moving stars stretch both axes over hundreds of units, which
squeezes everything else into a corner; zoom both at once and the cluster stands out as a knot.
The transparency is measured again from whatever is left inside the window, so a close zoom does
not come out as faint as the whole field.

Doing this before converting anything is deliberate. These five columns are what Gaia actually
measured, so nearly every star has them; the positions and velocities you compute in the next
step have gaps wherever a measurement was missing.

**4. Now work out where the stars really are.** This turns sky position, distance and motion
into real positions in space (`X`, `Y`, `Z`, in parsec) and real velocities (`U`, `V`, `W`, in
km/s), so you can look at your selection from any angle:

```python
stars = convert_to_galactic(stars)

%matplotlib widget

plot_3d(stars, "X", "Y", "Z", title="My selection in space",
        xlabel="X (pc)", ylabel="Y (pc)", zlabel="Z (pc)", color_col="my_cluster")
```

The `%matplotlib widget` line is what makes the plot interactive — without it, you get a still
image you cannot rotate. Does your selection still hold together in depth? On the sky it had to
look compact; that was your selection criterion, after all.

This plot fades its points too, and depth makes that matter more than it does in two
dimensions: solid points hide whatever sits behind them, so a whole field comes out as one
filled shape, while faded ones let the cluster show through as a darker knot inside it. Pass
`alpha=1` to see what it would look like otherwise.

Expect more `NaN` here than in step 2: Gaia measured a radial velocity for only a fraction of
its stars, and without it there is no `U`, `V`, `W`.

**5. Let the computer try — a different set of columns each time.** The algorithm only ever
looks at the columns you hand it. So run it several times, giving it something different to
work with each time, and give every run its own name so the results sit next to each other:

```python
# only where the stars sit on the sky -- what a picture of the sky shows
find_one_cluster(stars, columns=["ra", "dec"], cluster_label_column="sky")

# their real position in space, with the distance as a third axis
find_one_cluster(stars, columns=["X", "Y", "Z"], cluster_label_column="position")

# everything at once: the full six-dimensional phase space
find_one_cluster(stars, columns=["X", "Y", "Z", "U", "V", "W"], cluster_label_column="phase_space")
```

The knobs — `min_cluster_size` for HDBSCAN, `min_samples` and `eps` for DBSCAN — are left out
here on purpose. Given no value, each is measured from the data you handed over and printed, so
that the same call works on a small test table and on a catalogue of tens of thousands of stars.
`eps` in particular cannot have a fixed default: it is a length in the units of the columns you
chose, and half a degree on the sky is nothing like half a parsec in `X, Y, Z`. Treat the printed
values as a starting point and change them — that is the whole exercise.

Two things the code says out loud, because both look like an answer otherwise. If the group it
found covers a large part of your stars, it is the field itself rather than a cluster: in those
columns the stars did not separate. And if the group holds *exactly* `min_cluster_size` stars,
that number came from your setting rather than from the data — run it again with a different one
and see whether the group changes size with it.

Each run adds one column — `sky_HDBSCAN`, `position_HDBSCAN`, `phase_space_HDBSCAN`. Every
star gets a number in it: `0`, `1`, `2`, … for the groups that were found, and **`-1` for stars
that belong to no group at all**. If you forget to give a run its own name, the next run
overwrites the previous one; the code warns you when that is about to happen.

**6. Compare the runs and sharpen your selection.** This is the interesting part. The three runs
will not agree, and the disagreements tell you something:

```python
plot_3d(
    stars,
    "X",
    "Y",
    "Z",
    title="Grouped on the sky alone, shown by where the stars really are",
    xlabel="X (pc)",
    ylabel="Y (pc)",
    zlabel="Z (pc)",
    color_col="sky_HDBSCAN",
)
```

Plot every run the same way you selected by hand: first the plain view, then the same view
coloured by the result. Without the first picture you cannot tell whether the algorithm found
the structure you would have picked out yourself.

Each group is drawn in its own colour **and** its own marker shape, and the histogram hatches
its bars the same way. That is deliberate: a picture that separates groups by colour alone falls
apart in a greyscale handout and for anyone with a colour vision deficiency, so the shape repeats
whatever the colour says.

Swap `color_col` for `"position_HDBSCAN"` or `"phase_space_HDBSCAN"` and look again. Where do
the three ways of grouping pick out the same stars, and where do they part company? Use what you
see to go back to step 3, adjust your own selection, and run step 5 again. Real analysis works
exactly like this — a few rounds, each one a bit better informed than the last. Stars labelled
`-1` are drawn in grey.

Two runs worth adding if you have time: `["pmra", "pmdec"]`, the proper motion straight from
the catalogue, and `["U", "V", "W"]`, the real velocities. Draw the second one in `X`, `Y`, `Z`
— grouped by motion, shown by position. If the coloured stars sit together in space as well,
two independent measurements agree with each other, which is a much stronger argument than
either alone. Run the proper motion through all three — `mode="HDBSCAN"`, `mode="DBSCAN"` and
`mode="GMM"`. The field stars there form a broad spread rather than a clump with an edge, and
which of the two density-based methods copes with that depends on the cluster. The mixture
model is the interesting third case: it lays `n_cluster` bell curves over the data and gives
every star the one it fits best, so this is where you can watch that assumption meet data that
does not obey it. Look at where the boundary between its groups falls, and note that
`n_cluster` is the one knob nothing can measure for you.

**7. How old is it?** The stars of a cluster were born together, so in a plot of
colour against brightness they lie on one line — and that line changes shape as the cluster
ages, because heavy stars burn out first. Model lines of a given age are called isochrones;
download them from the [CMD form](http://stev.oapd.inaf.it/cmd) with the photometric system set
to Gaia DR2 (Evans et al. 2018) and a wide range of ages, then lay them over your own stars.

Two things have to line up first. Your stars need their distance divided out, so that their
brightness can be compared with a model. And the y-axis has to be flipped, because a magnitude
counts backwards — the smaller the number, the brighter the star.

```python
cluster = add_absolute_magnitude(stars[stars["my_selection"]])
isochrones = load_isochrones("data/isochrones.dat")

for age in [40, 100, 250, 1000, 2500]:  # millions of years
    line = get_isochrone(isochrones, age)
    plot_and_save(
        cluster,
        "bp_rp",
        "abs_g_mag",
        title=f"{age} million years",
        xlabel="BP - RP",
        ylabel="absolute G",
        invert_yaxis=True,
        line_x=line["bp_rp"],
        line_y=line["Gmag"],
        line_label=f"{age} Myr",
    )
```

This needs two columns from your Gaia download that nothing else used: `phot_g_mean_mag` and
`bp_rp`. The youngest lines will sit above your stars at the faint red end, the oldest will bend
away far too early — between those two you have bracketed the age without having known it
beforehand. How tightly you can bracket it depends on how many bright stars you have, and that
is worth thinking about rather than glossing over.

### Things to try

There is no single right answer here, and you are not trying to find one. The point is to get
a feel for what each method can and cannot see.

- Which measurement separates the group most clearly — distance, sky position, or proper
  motion? Why might that be?
- Change `min_cluster_size` in step 5, starting from the value that was printed for you. What
  happens to the number of stars in your group, and at what point does the answer stop making
  sense? Is there a range where it barely changes at all — and how much more would you trust a
  value from the middle of that range than one from its edge?
- Compare the three algorithms: `mode="HDBSCAN"`, `mode="DBSCAN"` and `mode="GMM"` each add
  their own column, so you can plot them side by side. Where do they disagree?
- Each of the three runs sees more than the one before it: the sky, then the sky plus the
  distance, then everything including the motion. Does each step actually improve the result?
- The phase space run has all six columns and so the most information of the three. Does it
  give the best answer? Look at the spread of `X, Y, Z` against that of `U, V, W` before you
  decide why.
- How many stars did you pick by hand in step 3, and how many did the algorithm find? Look at
  the ones you disagree about — who do you think is right, and how could you check?

### The functions

| Function | What it does |
| --- | --- |
| `load_fits(filename)` | Read a FITS table (a Gaia download) into a table |
| `load_parquet(filename)` | Read a Parquet file into a table |
| `save_parquet(dataframe, filename)` | Save a table as a Parquet file |
| `add_distance(df)` | Add the distance `distance_pc` from the parallax |
| `convert_to_galactic(df)` | Add positions `X, Y, Z` and velocities `U, V, W` |
| `plot_histogram(df, column)` | Show how one quantity is distributed, with `xlim` to zoom in |
| `plot_dataframe(df, x_col, y_col, …)` | Build a scatter plot and hand back the figure |
| `plot_and_save(df, x_col, y_col, …)` | Scatter plot, optionally with an ellipse or a line |
| `plot_ellipse(params)` | Draw an ellipse onto the plot you just made |
| `plot_3d(df, x_col, y_col, z_col, …)` | Scatter plot in three dimensions |
| `select_range(df, column, minimum, maximum)` | Mark the stars between two limits and report how many |
| `select_ellipse(df, x_col, y_col, params)` | Mark the stars inside an ellipse you drew and report how many |
| `find_one_cluster(df, columns)` | Find clusters with HDBSCAN, DBSCAN or a Gaussian Mixture Model |
| `add_absolute_magnitude(df)` | Add `abs_g_mag`, the brightness with the distance divided out |
| `load_isochrones(filename)` | Read a PARSEC/CMD isochrone file into a table |
| `get_isochrone(isochrones, age_myr)` | Pick the isochrone of one age, ready to draw as a line |

Every function explains its arguments in its own docstring. In a notebook, run
`help(find_one_cluster)` or put a `?` after the name to read it.

### What this has been tested on

Everything below was run against real data, so it is worth saying plainly what that data was
and where the approach holds up. All of it is Gaia DR3, a 4° cone, `parallax BETWEEN 3 AND 12`:

| Cluster | Centre (ra, dec) | Stars |
| --- | --- | --- |
| Pleiades (Melotte 22) | 56.60, +24.11 | 15,540 |
| Praesepe / Beehive (NGC 2632) | 130.05, +19.62 | 12,216 |

The isochrones were PARSEC models from the [CMD 3.9 form](http://stev.oapd.inaf.it/cmd), Gaia DR2
photometry from Evans et al. 2018, solar metallicity, `log(age/yr)` from 7.0 to 9.6 in steps of 0.2 — 14 ages between
10 million and 4 billion years.

**What works on both clusters.** Selecting by hand, throughout: none of the starting values in
notebook 1 are cluster-specific, they are all read off the data. Clustering on `X, Y, Z` with
HDBSCAN. Clustering on proper motion with DBSCAN. Clustering on `U, V, W`, with the caveat that
only about a fifth of the stars have a radial velocity at all, so that run uses far fewer stars
than the others.

**What does not, and why it is worth knowing.**

- **Clustering on `ra, dec` returns the whole field, on both clusters.** This is deliberate and
  the notebook builds on it: on the sky a cluster is a mild overdensity rather than an island,
  and a density-based algorithm has no gap to find. The code warns you when it happens.
- **HDBSCAN on proper motion works for the Pleiades and fails for Praesepe.** The Pleiades sit
  about 1.0σ clear of the field's proper-motion spread, Praesepe only 0.8σ, and below roughly
  that point HDBSCAN returns the field instead. DBSCAN finds both. If your own cluster's motion
  is not distinctive, expect the same.
- **The six-dimensional run returns exactly `min_cluster_size` stars on both clusters** — the
  tip of a single density peak rather than a cluster. It warns, and the number is an artefact of
  the setting, not a measurement.
- **The Pleiades cannot be dated precisely, and this is a property of the data.** Gaia saturates
  above about `G = 3`, so the brightest members are missing and the main-sequence turn-off — the
  only part of the diagram that is strongly age-sensitive — has almost no stars in it. The
  Pleiades can be bracketed to very roughly 60–250 million years against a literature value of
  125. Praesepe, older and so turning off where Gaia measures well, narrows to between 631 and
  1000 million years against a literature value near 700–800. An older cluster dates better.

**Not tested:** any other cluster, any survey other than Gaia DR3, isochrones in another
photometric system or at non-solar metallicity, and downloads without a parallax cut. The code
should cope with all of those, but nobody has checked.

---

## For instructors and maintainers

The student walkthrough above also exists in German as
[`README.de.md`](README.de.md). The two are translations of each other — when you change one,
change the other, or the group working in German will follow outdated steps.

### Layout

The package lives in `src/stellar_cluster_finder/` and is installed editable by `uv sync`:

| Module | Contents |
| --- | --- |
| `files.py` | FITS and Parquet loading and saving |
| `coordinates.py` | Parallax → distance, and ICRS → Galactic conversion via astropy |
| `plotting.py` | 1D, 2D and 3D plotting on matplotlib/seaborn |
| `selection.py` | Turning histogram limits and drawn ellipses into True/False columns |
| `clustering.py` | HDBSCAN, DBSCAN and Gaussian Mixture Model wrappers |
| `isochrones.py` | PARSEC isochrone files, absolute magnitudes, cluster ages |

`main.py` in the repository root is a "hello world" stub kept for demonstration purposes.
Generated output (sample tables, example figures) goes to `examples/` and is gitignored.

### Checks

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run ty check         # type check
```

**There is deliberately no `tests/` directory.** A separate test suite would roughly double
the amount of code students have to look past, so each module instead carries its own checks
in its `if __name__ == "__main__"` block: it builds synthetic data, calls the module's
functions and asserts the invariants that matter. Run all six after any change — a broken
module exits non-zero:

```bash
for m in files coordinates selection clustering plotting isochrones; do
    uv run python src/stellar_cluster_finder/$m.py
done
```

Call them by file path rather than `python -m stellar_cluster_finder.<module>`; `__init__.py`
imports every submodule, so the `-m` form produces a harmless but confusing `runpy` warning.
