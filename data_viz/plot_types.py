
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('default')
from typing import Tuple

def grid_plot(grid_subset: pd.DataFrame,
              seaborn_func: callable,
              rows: int,
              cols: int,
              group_var: str,
              figsize: tuple = (10, 5),
              legend_loc: str = "lower",
              **kwargs):
    facet = grid_subset[group_var].unique()
    facet = facet.reshape(rows, cols)
    fig, axes = plt.subplots(
        facet.shape[0], facet.shape[1], sharex=True, sharey=True, figsize=figsize)
    if rows > 1:
        for m in range(facet.shape[0]):
            for n in range(facet.shape[1]):
                group = facet[m, n]
                subset = grid_subset[grid_subset[group_var] == group]
                if legend_loc == 'lower':
                    legend = True if (m+1 == rows) and (n+1 == cols) else False
                elif legend_loc == "upper":
                    legend = True if (m+1 == 1) and (n+1 == cols) else False
                else:
                    legend = False
                seaborn_func(
                    data=subset,
                    ax=axes[m, n],
                    legend=legend,
                    **kwargs)
                axes[m, n].set(ylabel=None, xlabel=None, title=group)
                axes[m, n].grid()
    else:
        for n in range(facet.shape[1]):
            group = facet[n]
            subset = grid_subset[grid_subset[group_var] == group]
            if legend_loc == 'lower':
                legend = True if (m+1 == rows) and (n+1 == cols) else False
            elif legend_loc == "upper":
                legend = True if (m+1 == 1) and (n+1 == cols) else False
            else:
                legend = False
            seaborn_func(
                data=subset,
                ax=axes[m, n],
                legend=legend,
                **kwargs)
            axes[n].set(ylabel=None, xlabel=None, title=group)
            axes[n].grid()
    return fig, axes


def time_overlay_plot(month_time_series: pd.Series,
                      ax: matplotlib.axes.Axes,
                      step_size: int,
                      highlight_year: int,
                      date_formatter: str,
                      bg_alpha: float = .3) -> Tuple[matplotlib.axes.Axes, pd.Index]:
    """
    Create an overlay plot where the x-axis represents months and each series corresponds to a year of data.

    This function reads in a month-level time series with data spanning multiple years. It overlays the data for each year 
    on the same plot, allowing for year-over-year comparison. One year can be highlighted for emphasis.

    Args:
        month_time_series (pd.Series): A time series with observations at one-month intervals.
        ax (matplotlib.axes.Axes): A Matplotlib axis object to plot on.
        step_size (int): The number of years to skip between plotted series.
        highlight_year (int): The year to highlight in the overlay plot. Set to None for no highlight.
        date_formatter (str): The format string for the date labels on the x-axis.
        bg_alpha (float): The alpha transparency for the non-highlighted lines. Defaults to 0.3.

    Returns:
        matplotlib.axes.Axes: The updated axis object with the overlay plot.
        pd.Index: The x-tick labels of the plot.

    Raises:
        AssertionError: If step_size is larger than the number of unique years in the time series.
    """
    target_years = month_time_series.index.year.unique()[::step_size]
    assert (len(target_years) > step_size), "Step size larger than index length"
    labs = month_time_series.loc[str(
        target_years[-step_size]):str(target_years[-1])].index.strftime(date_formatter)
    for i in range(len(target_years)-(step_size-1)):
        a = 1 if (target_years[i + (step_size-1)]
                  == highlight_year) else bg_alpha
        subset_time_series = (
            month_time_series
            .loc[str(target_years[i]): str(target_years[i + (step_size-1)])]
            .reset_index()
            .set_index(labs)[month_time_series.name]
        )
        label = fr"${target_years[i]} \rightarrow {target_years[i + (step_size-1)]}$" if step_size != 1 else fr"${target_years[i]}$"
        subset_time_series.plot(linewidth=3,
                                ax=ax,
                                alpha=a,
                                label=label)
    return ax, labs