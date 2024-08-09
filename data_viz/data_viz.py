import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
plt.style.use('default')
from datetime import datetime
from typing import Tuple, Union
from collections.abc import Iterable


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



def custom_legend(ax: matplotlib.axes.Axes,
                  outside_loc: str = None,
                  order: Union[str, list] = "default",
                  title: str = "",
                  linewidth: int = 2,
                  **kwargs) -> matplotlib.axes.Axes:
    """
    Customize the legend location and order on a Matplotlib axis.

    This function adjusts the position of the legend, optionally placing it outside the plot area,
    and can reorder the legend entries based on specified criteria.

    Args:
        ax (matplotlib.axes.Axes): An existing Matplotlib axis object to which the legend belongs.
        outside_loc (str, optional): Specifies the location of the legend outside the plot area.
                                     Must be one of ["lower", "center", "upper", None]. 
                                     Defaults to None.
        order (str, optional): Determines the order of the legend entries. 
                               Must be one of ["default", "reverse", "desc"]. 
                               "default" keeps the current order, 
                               "reverse" reverses the current order, 
                               "desc" orders entries by descending values. 
                               Defaults to "default".

    Returns:
        matplotlib.axes.Axes: The axis object with the customized legend.

    Raises:
        AssertionError: If `outside_loc` is not in ["lower", "center", "upper", None].
    """
    handles, labels = ax.get_legend_handles_labels()
    if order == 'default':
        pass
    elif order == 'reverse':
        handles = handles[::-1]
        labels = labels[::-1]
    elif order == 'desc':
        ordering = np.flip(np.argsort(np.array([line.get_ydata()[-1] for line in ax.lines if (len(line.get_ydata())>0 and line.get_label() in labels)])))
        handles = np.array(handles)[ordering].tolist()
        labels = np.array(labels)[ordering].tolist()
    elif isinstance(order, Iterable):
        value_to_index = {value: idx for idx, value in enumerate(labels)}
        indices = [value_to_index[value] for value in order]
        labels = list(order)
        handles = np.array(handles)[indices].tolist()
    else:
        raise Exception("Invalid Order")
    error_msg = "legend_to_right loc must be None or in 'lower', 'center', or 'upper'"
    assert outside_loc in ["lower", "center", "upper", None], error_msg
    if outside_loc == "lower":
        ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(1, 0), **kwargs)
    elif outside_loc == "center":
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, .5), **kwargs)
    elif outside_loc == "upper":
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), **kwargs)
    else:
        ax.legend(handles, labels, **kwargs)
    legend = ax.get_legend()
    legend.set_title(title)
    for line in legend.get_lines():
        line.set_linewidth(linewidth)
    return ax


def build_colormap(series: pd.Series) -> dict:
    """
    Build a colormap dictionary for a pandas Series.

    This function creates a dictionary that maps each unique value in the input
    Series to a unique color from the 'tab10' colormap provided by Seaborn. The 
    function ensures that the number of unique values in the Series is less than 10.

    Parameters:
    ----------
    series : pd.Series
        The pandas Series for which to build the colormap. Each unique value in the 
        Series will be assigned a unique color.

    Returns:
    -------
    dict
        A dictionary where the keys are the unique values from the input Series 
        and the values are the corresponding colors from the 'tab10' colormap.

    Raises:
    ------
    AssertionError
        If the number of unique values in the Series is 10 or more.
    """
    unique_set = series.unique()
    assert len(unique_set) < 10
    colors = sns.color_palette("tab10")
    colormap = {}
    for d, c in zip(unique_set, colors[:len(unique_set)]):
        colormap[d] = c
    return colormap


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