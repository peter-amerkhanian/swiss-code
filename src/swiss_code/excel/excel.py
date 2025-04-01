import os
import xlwings as xw
import string
import pandas as pd
import numpy as np
import pywintypes
from typing import Optional

def get_or_create_workbook(
    filename: str, display_alerts: bool = False, screen_updating: bool = False
) -> xw.Book:
    """
    Checks if the specified Excel file exists. If it does, opens it;
    otherwise, creates a new one. Returns the workbook object.

    Args:
        filename (str): The name of the Excel file. Default is "output.xlsx".

    Returns:
        xlwings.Book: The opened or newly created workbook.
    """
    if os.path.exists(filename):
        wb = xw.Book(filename)
        wb.app.display_alerts = display_alerts
        wb.app.screen_updating = screen_updating
    else:
        wb = xw.Book()
        wb.app.display_alerts = display_alerts
        wb.app.screen_updating = screen_updating
        wb.save(filename)
    return wb


def select_sheet(name: str, wb: xw.Book) -> xw.Sheet:
    """
    Selects an existing sheet by name or creates a new one if it does not exist.

    Args:
        name (str): The name of the sheet to select or create.
        wb (xlwings.Book): The Excel workbook object.

    Returns:
        xlwings.Sheet: The selected or newly created sheet."
    """
    try:
        sheet_new = wb.sheets.add(name)
    except ValueError:
        sheet_new = wb.sheets[name]
    return sheet_new


def get_data_bounds(sheet):
    start_col = string.ascii_uppercase[sheet.used_range[0].column - 1]
    end_col = string.ascii_uppercase[sheet.used_range[-1].column - 1]
    start_row = sheet.used_range[0].row
    end_row = sheet.used_range[-1].row
    return {
        "start_col": start_col,
        "end_col": end_col,
        "start_row": start_row,
        "end_row": end_row,
    }


excel_colors = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "dark_red": (192, 0, 0),
    "blue": (0, 0, 255),
    "dark_blue": (0, 0, 128),
    "green": (0, 255, 0),
    "dark_green": (0, 128, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "grey": (128, 128, 128),
    "light_grey": (217, 217, 217),  # Default Excel light grey
    "dark_grey": (64, 64, 64),
    "light_blue": (221, 235, 247),
    "light_green": (198, 224, 180),
    "light_yellow": (255, 242, 204),
    "light_orange": (255, 229, 153),
    "light_purple": (204, 192, 218),
    "light_red": (252, 228, 214),
    "teal": (0, 128, 128),
    "gold": (255, 192, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}

excel_h_align = {
    "Center": -4108,
    "Center across selection": 7,
    "Distribute": -4117,
    "Fill": 5,
    "Align according to data type": 1,
    "Justify": -4130,
    "Left": -4131,
    "Right": -4152,
}

excel_v_align = {
    "Bottom": -4107,
    "Center": -4108,
    "Distributed": -4117,
    "Justify": -4130,
    "Top": -4160,
}


class ExcelDataFrame:
    def __init__(self, df: pd.DataFrame, sheet: xw.Sheet, range: xw.Range):
        """
        Initializes an ExcelDataFrame object.

        :param df: Pandas DataFrame containing the data
        :param sheet: xlwings Sheet object where data will be stored
        :param range_: xlwings Range object representing where the data is placed
        """
        self.df = df
        self.sheet = sheet
        self.range = range
        self.index_start_row = range[0].row
        self.index_end_row = range[-1].row
        self.header_start_col = range[0].address.split("$")[1]
        self.header_end_col = range[-1].address.split("$")[1]

    def make_borders(self, linestyle=1, weight=2):
        """
        Applies borders to a given range in the Excel sheet.

        Args:
            data_range (xlwings.Range): The range to apply borders to.
            linestyle (int, optional): Line style for borders. Default is 1.
            weight (int, optional): Border weight. Default is 2.
        """
        self.range.api.Borders.LineStyle = linestyle
        self.range.api.Borders.Weight = weight

    def merge_axis(self, index=1, axis=1):
        """
        Merges adjacent cells in the specified column if they have the same value.

        Args:
            col (int): The column number (1-based) to check for merging. Default is the first column.
        """
        if axis == 1:
            last_cell = self.range.last_cell.row
        elif axis == 0:
            last_cell = self.range.last_cell.column
        else:
            raise ValueError

        prev_value, merge_start = None, None
        for free_cell in range(1, last_cell + 1):
            if axis == 1:
                cell_value = self.sheet.cells(free_cell, index).value
            else:
                cell_value = self.sheet.cells(index, free_cell).value
            if cell_value == prev_value:
                continue
            else:
                if (
                    prev_value is not None
                    and merge_start is not None
                    and free_cell - merge_start > 1
                ):
                    if axis == 1:
                        self.sheet.range(
                            (merge_start, index), (free_cell - 1, index)
                        ).api.Merge()
                    else:
                        self.sheet.range(
                            (index, merge_start), (index, free_cell - 1)
                        ).api.Merge()
                prev_value = cell_value
                merge_start = free_cell

        if (
            prev_value is not None
            and merge_start is not None
            and last_cell - merge_start > 0
        ):
            if axis == 1:
                self.sheet.range((merge_start, index), (last_cell, index)).api.Merge()
            else:
                self.sheet.range((index, merge_start), (index, last_cell)).api.Merge()

    def format_column_data(self, header_name: str, format: str = "$#,###.00"):
        """
        Formats the column with the given header name as a dollar amount in an Excel sheet.

        Args:
            df_range (dict): Dictionary containing DataFrame and range information.
            header_name (str): The column header to search for.
            format (str, optional): The Excel number format. Default is "$#,###.00".
        """
        if isinstance(self.df.columns, pd.MultiIndex):
            for ind, col in enumerate(self.df.columns):
                if col == header_name:
                    col_index = self.df.index.nlevels + ind + 1
                    break
        else:
            headers = self.range[0].expand("right").value
            if header_name not in headers:
                raise ValueError(f"Header '{header_name}' not found in the sheet.")

            col_index = headers.index(header_name) + 1
        col_letter = xw.utils.col_name(col_index)

        data_start = self.df.columns.nlevels + 1
        data_end = data_start + self.df.shape[0] - 1
        self.sheet.range(
            f"{col_letter}{data_start}:{col_letter}{data_end}"
        ).number_format = format

    def _format_range(
        self,
        data_range: xw.Range,
        bold: bool = False,
        color: Optional[str] = None,
        h_align: Optional[str] = None,
        v_align: Optional[str] = None,
    ) -> None:
        """
        Applies formatting to a given range in Excel.

        Args:
            data_range (xw.Range): The range to format.
            bold (bool, optional): Whether to bold the text.
            color (Optional[str], optional): The fill color (from excel_colors dictionary).
            h_align (Optional[str], optional): Horizontal alignment (from excel_h_align dictionary).
            v_align (Optional[str], optional): Vertical alignment (from excel_v_align dictionary).
        """
        if bold != "ignore":
            data_range.font.bold = bold
        if color != "ignore":
            data_range.color = excel_colors.get(color)
        if h_align != "ignore":
            data_range.api.HorizontalAlignment = excel_h_align.get(h_align)
        if v_align != "ignore":
            data_range.api.VerticalAlignment = excel_v_align.get(v_align)

    def format_column(
        self, col_name, bold=False, color=None, h_align=None, v_align=None
    ):
        if isinstance(self.df.columns, pd.MultiIndex):
            for ind, col in enumerate(self.df.columns):
                if col == col_name:
                    col_index = self.df.index.nlevels + ind + 1
                    break
        else:
            headers = self.range[0].expand("right").value
            if col_name not in headers:
                raise ValueError(f"Header '{col_name}' not found in the sheet.")
            col_index = headers.index(col_name) + 1
        col_letter = xw.utils.col_name(col_index)
        data_col_range = self.sheet.range(
            f"{col_letter}{self.index_start_row}:{col_letter}{self.index_end_row}"
        )
        self._format_range(data_col_range, bold, color, h_align, v_align)

    def format_row(self, row_name, bold=False, color=None, h_align=None, v_align=None):
        if isinstance(self.df.index, pd.MultiIndex):
            for ind, indexer in enumerate(self.df.index):
                if indexer == row_name:
                    data_row_ind = self.df.columns.nlevels + ind + 1
                    break
        else:
            index = self.range[0].expand("down").value
            if row_name not in index:
                raise ValueError(f"Row '{row_name}' not found in the sheet.")
            data_row_ind = index.index(row_name) + 1
        data_row_range = self.sheet.range(
            f"{self.header_start_col}{data_row_ind}:{self.header_end_col}{data_row_ind}"
        )
        self._format_range(data_row_range, bold, color, h_align, v_align)

    def format_indices(self, bold=False, color=None, h_align=None, v_align=None):
        """
        Bolds index and column headers in an Excel sheet.
        """
        if isinstance(self.df.index, pd.MultiIndex):
            for i in range(self.df.index.nlevels):
                level_range = self.sheet.range(
                    f"{string.ascii_uppercase[i]}{self.index_start_row}:{string.ascii_uppercase[i]}{self.index_end_row}"
                )
                self._format_range(level_range, bold, color, h_align, v_align)
        else:
            index_range = self.sheet.range(
                f"{self.header_start_col}{self.index_start_row}:{self.header_start_col}{self.index_end_row}"
            )
            self._format_range(index_range, bold, color, h_align, v_align)

        if isinstance(self.df.columns, pd.MultiIndex):
            for i in range(self.df.columns.nlevels):
                level_range = self.sheet.range(
                    f"{self.header_start_col}{i+1}:{self.header_end_col}{i+1}"
                )
                self._format_range(level_range, bold, color, h_align, v_align)
        else:
            header_range = self.sheet.range(
                f"{self.header_start_col}{self.index_start_row}:{self.header_end_col}{self.index_start_row}"
            )
            self._format_range(header_range, bold, color, h_align, v_align)

    def __repr__(self):
        try:
            return f"ACTIVE ExcelDataFrame(sheet={self.sheet.name}, range={self.range.address}, df_shape={self.df.shape})"
        except pywintypes.com_error:
            return f" DISCONNECTED ExcelDataFrame(sheet=NA, range=NA, df_shape={self.df.shape})"


def get_df_range(df: pd.DataFrame) -> tuple[int, int]:
    base_shape: np.ndarray = np.array(df.shape, dtype=int)
    columns: int = df.columns.nlevels
    indices: int = df.index.nlevels
    shape = base_shape + np.array([columns, indices])
    return shape


def write_df_to_excel(
    df: pd.DataFrame, sheet: xw.Sheet, cell_start: str = "A1"
) -> ExcelDataFrame:
    """
    Writes a pandas DataFrame to an Excel sheet and bolds the index and column headers.

    Args:
        df (pd.DataFrame): The DataFrame to write.
        sheet (xlwings.Sheet): The Excel sheet where the DataFrame will be written.
    """
    # Write the DataFrame to the Excel sheet starting from cell A1
    sheet.range(cell_start).options(index=True, header=True).value = df
    df_range = get_df_range(df)
    end_col = df_range[1] - 1
    end_row = df_range[0]
    return ExcelDataFrame(
        df, sheet, xw.Range(f"{cell_start}:{string.ascii_uppercase[end_col]}{end_row}")
    )


def autofit_all_sheets(wb: xw.Book):
    """
    Autofits all columns in all sheets of the given workbook.

    Args:
        wb (xlwings.Book): The Excel workbook object.
    """
    for sheet in wb.sheets:
        if sheet.used_range.columns.count > 1:  # Ensure there's data in the sheet
            sheet.used_range.api.EntireColumn.AutoFit()  # Autofit columns
            sheet.used_range.api.EntireRow.AutoFit()  # Autofit rows


def close_out_book(wb: xw.Book, autofit: bool = True):
    if autofit:
        autofit_all_sheets(wb)
    if "Sheet1" in [sheet.name for sheet in wb.sheets]:
        wb.sheets["Sheet1"].delete()
    wb.save()
    wb.app.quit()
