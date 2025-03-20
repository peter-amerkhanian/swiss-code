import os
import xlwings as xw
import string
import pandas as pd
import numpy as np
import pywintypes

def get_or_create_workbook(filename: str,
                           display_alerts: bool = False,
                           screen_updating: bool= False) -> xw.Book:
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
        'start_col': start_col,
        'end_col': end_col,
        'start_row': start_row,
        'end_row': end_row
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
    "magenta": (255, 0, 255)
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
    
    def merge_row(self, row=1):
        """
        Merges adjacent cells in the specified row if they have the same value.

        Args:
            row (int): The row number (1-based) to check for merging. Default is the first row.
        """
        last_col = self.sheet.range(row, self.sheet.cells.last_cell.column).end("left").column
        prev_value, merge_start = None, None

        for col in range(1, last_col + 1):
            cell_value = self.sheet.cells(row, col).value

            if cell_value == prev_value:
                continue
            else:
                if prev_value is not None and merge_start is not None and col - merge_start > 1:
                    self.sheet.range((row, merge_start), (row, col - 1)).api.Merge()
                prev_value = cell_value
                merge_start = col

        if prev_value is not None and merge_start is not None and last_col - merge_start > 0:
            self.sheet.range((row, merge_start), (row, last_col)).api.Merge()

    def merge_column(self, col=1):
        """
        Merges adjacent cells in the specified column if they have the same value.

        Args:
            col (int): The column number (1-based) to check for merging. Default is the first column.
        """
        last_row = self.sheet.range(self.sheet.cells.last_cell.row, col).end("up").row
        prev_value, merge_start = None, None

        for row in range(1, last_row + 1):
            cell_value = self.sheet.cells(row, col).value

            if cell_value == prev_value:
                continue
            else:
                if prev_value is not None and merge_start is not None and row - merge_start > 1:
                    self.sheet.range((merge_start, col), (row - 1, col)).api.Merge()
                prev_value = cell_value
                merge_start = row

        if prev_value is not None and merge_start is not None and last_row - merge_start > 0:
            self.sheet.range((merge_start, col), (last_row, col)).api.Merge()

    def number_format_column(self, header_name: str, format:str ="$#,###.00"):
        """
        Formats the column with the given header name as a dollar amount in an Excel sheet.

        Args:
            df_range (dict): Dictionary containing DataFrame and range information.
            header_name (str): The column header to search for.
            format (str, optional): The Excel number format. Default is "$#,###.00".
        """
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

    def format_column(self, col_name, bold=False, color=None):
        headers = self.range[0].expand('right').value
        if col_name not in headers:
            raise ValueError(f"Header '{col_name}' not found in the sheet.")
        col_index = headers.index(col_name) + 1
        col_letter = xw.utils.col_name(col_index)
        data_col_range = self.sheet.range(f"{col_letter}{self.index_start_row}:{col_letter}{self.index_end_row}")
        if not bold == "ignore":
            data_col_range.font.bold = bold
        if not color == 'ignore':
            data_col_range.color = excel_colors.get(color)

    def format_row(self, row_name, bold=False, color=None):
        index = self.range[0].expand('down').value
        if row_name not in index:
            raise ValueError(f"Row '{row_name}' not found in the sheet.")
        data_row_ind = index.index(row_name) + 1
        data_row_range = self.sheet.range(f"{self.header_start_col}{data_row_ind}:{self.header_end_col}{data_row_ind}")
        if not bold == "ignore":
            data_row_range.font.bold = bold
        if not color == 'ignore':
            data_row_range.color = excel_colors.get(color)

    def format_indices(self, bold=False, color=None):
        """
        Bolds index and column headers in an Excel sheet.
        """
        if isinstance(self.df.index, pd.MultiIndex):
            for i in range(self.df.index.nlevels):
                level_range = self.sheet.range(f"{string.ascii_uppercase[i]}{self.index_start_row}:{string.ascii_uppercase[i]}{self.index_end_row}")
                if not bold == "ignore":
                    level_range.font.bold = bold
                if not color == 'ignore':
                    level_range.color = excel_colors.get(color)
        else:
            index_range = self.sheet.range(f"{self.header_start_col}{self.index_start_row}:{self.header_start_col}{self.index_end_row}")
            if not bold == "ignore":
                index_range.font.bold = bold
            if not color == 'ignore':
                index_range.color = excel_colors.get(color)
        
        if isinstance(self.df.columns, pd.MultiIndex):
            for i in range(self.df.columns.nlevels):
                level_range = self.sheet.range(f"{self.header_start_col}{i+1}:{self.header_end_col}{i+1}")
                if not bold == "ignore":
                    level_range.font.bold = bold
                if not color == 'ignore':
                    level_range.color = excel_colors.get(color)
        else:
            header_range = self.sheet.range(f"{self.header_start_col}{self.index_start_row}:{self.header_end_col}{self.index_start_row}")
            if not bold == "ignore":
                header_range.font.bold = bold
            if not color == 'ignore':
                header_range.color = excel_colors.get(color)

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

def write_df_to_excel(df: pd.DataFrame,
                      sheet: xw.Sheet,
                      cell_start: str="A1") -> ExcelDataFrame:
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
    return ExcelDataFrame(df, sheet, xw.Range(f"{cell_start}:{string.ascii_uppercase[end_col]}{end_row}"))
        


def autofit_all_sheets(wb: xw.Book):
    """
    Autofits all columns in all sheets of the given workbook.
    
    Args:
        wb (xlwings.Book): The Excel workbook object.
    """
    for sheet in wb.sheets:
        if sheet.used_range.columns.count > 1:  # Ensure there's data in the sheet
            sheet.used_range.api.EntireColumn.AutoFit()  # Autofit columns
            sheet.used_range.api.EntireRow.AutoFit()     # Autofit rows


def close_out_book(wb: xw.Book, autofit: bool=True):
    if autofit:
        autofit_all_sheets(wb)
    if "Sheet1" in [sheet.name for sheet in wb.sheets]:
        wb.sheets["Sheet1"].delete()
    wb.save()
    wb.app.quit()

def merge_row(sheet, row=1):
    """
    Merges adjacent cells in the specified row if they have the same value.

    Args:
        sheet (xlwings.Sheet): The Excel sheet where merging should occur.
        row (int): The row number (1-based) to check for merging. Default is the first row.
    """
    # Find the last used column in the specified row
    last_col = sheet.range(row, sheet.cells.last_cell.column).end("left").column
    
    prev_value, merge_start = None, None

    for col in range(1, last_col + 1):  # Iterate over all columns
        cell_value = sheet.cells(row, col).value

        if cell_value == prev_value:
            # Continue merging range
            continue
        else:
            # Merge previous range if applicable
            if prev_value is not None and merge_start is not None and col - merge_start > 1:
                sheet.range((row, merge_start), (row, col - 1)).api.Merge()

            # Start new merge group
            prev_value = cell_value
            merge_start = col

    # Merge last group (if applicable)
    if prev_value is not None and merge_start is not None and last_col - merge_start > 0:
        sheet.range((row, merge_start), (row, last_col)).api.Merge()


def merge_column(sheet, col=1):
    """
    Merges adjacent cells in the specified column if they have the same value.

    Args:
        sheet (xlwings.Sheet): The Excel sheet where merging should occur.
        col (int): The column number (1-based) to check for merging. Default is the first column.
    """
    # Find the last used row in the specified column
    last_row = sheet.range(sheet.cells.last_cell.row, col).end("up").row

    prev_value, merge_start = None, None

    for row in range(1, last_row + 1):  # Iterate over all rows
        cell_value = sheet.cells(row, col).value

        if cell_value == prev_value:
            # Continue merging range
            continue
        else:
            # Merge previous range if applicable
            if prev_value is not None and merge_start is not None and row - merge_start > 1:
                sheet.range((merge_start, col), (row - 1, col)).api.Merge()

            # Start new merge group
            prev_value = cell_value
            merge_start = row

    # Merge last group (if applicable)
    if prev_value is not None and merge_start is not None and last_row - merge_start > 0:
        sheet.range((merge_start, col), (last_row, col)).api.Merge()



def make_borders(data_range, linestyle=1, weight=2):
    data_range.api.Borders.LineStyle = linestyle
    data_range.api.Borders.Weight = weight
