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



def show_all_xticks(ax: matplotlib.axes.Axes, labs: pd.Index) -> matplotlib.axes.Axes:
    """
    Sets all x-ticks on a Matplotlib axis object and labels them with the provided list of labels.

    This function ensures that all x-ticks are displayed and labeled as specified, making it easier to read and interpret the x-axis of the plot. The labels are displayed horizontally.

    Parameters:
    ax (matplotlib.axes.Axes): The axis object on which to set the x-ticks and labels.
    labs (list of str): A list of labels to set on the x-axis. The number of labels should correspond to the number of ticks.

    Returns:
    matplotlib.axes.Axes: The modified axis object with updated x-ticks and labels.
    """
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=0)
    return ax

def comma_formatter() -> ticker.FuncFormatter:
    """
    Creates a custom Matplotlib tick formatter that formats axis ticks with commas.

    This formatter converts axis tick values to integers and formats them with commas for thousands 
    separators. This can be useful for improving the readability of plots with large numerical values.

    Returns:
    --------
    ticker.FuncFormatter:
        A Matplotlib FuncFormatter object that applies the comma formatting to axis ticks.
    """
    def comma(x: float, pos) -> str:
        """Format the tick value `x` with commas. The parameter `pos` is not used."""
        return '{:,}'.format(int(x))

    return ticker.FuncFormatter(comma)