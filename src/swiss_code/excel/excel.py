import os
import xlwings as xw
import string
import pandas as pd
import numpy as np


def get_or_create_workbook(filename: str,
                           display_alerts: bool = False,
                           screen_updating: bool= False):
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

def select_sheet(name: str, wb: xw.Book):
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


def bold_indices(df, sheet):
    # Get the range for the index
    index_start_row = 1  # Start from row 2 (since the header is in row 1)
    index_end_row = index_start_row + df.shape[0]
    # Get the range for the headers
    header_start_col = 1  # Start from column B (the first column is the index)
    header_end_col = header_start_col + df.reset_index().shape[1] - 1  # Adjust for headers

    # For multi-index columns, we need to get the full range
    if isinstance(df.index, pd.MultiIndex):
        for i in range(df.index.nlevels):
            # Define the range for each level of the multi-index
            level_range = sheet.range(f"{string.ascii_uppercase[i]}{index_start_row}:{string.ascii_uppercase[i]}{index_end_row}")
            level_range.font.bold = True
            index_end_row = index_start_row + df.shape[0] + 1
    else:
        index_range = sheet.range(f"A{index_start_row}:A{index_end_row}")
        index_range.font.bold = True
    if isinstance(df.columns, pd.MultiIndex):
        for i in range(df.columns.nlevels):
            # Define the range for each level of the multi-index
            level_range = sheet.range(f"B{i+1}:{string.ascii_uppercase[header_end_col]}{i+1}")
            level_range.font.bold = True
    else:
        # If it's a single index, bold the header range directly
        header_range = sheet.range(f"B1:{string.ascii_uppercase[header_end_col - 1]}1")
        header_range.font.bold = True


def write_df_to_excel(df: pd.DataFrame,
                      sheet: xw.Sheet,
                      cell_start: str="A1",
                      bold_indexes: bool=True):
    """
    Writes a pandas DataFrame to an Excel sheet and bolds the index and column headers.
    
    Args:
        df (pd.DataFrame): The DataFrame to write.
        sheet (xlwings.Sheet): The Excel sheet where the DataFrame will be written.
    """
    # Write the DataFrame to the Excel sheet starting from cell A1
    sheet.range(cell_start).options(index=True, header=True).value = df
    if bold_indexes:
        bold_indices(df, sheet)
        


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
    wb.close()

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


def format_percentage_column(sheet, header_name, format, header_row="A"):
    """
    Formats the column with the given header name as a three-digit percentage in an Excel sheet.
    
    Args:
        sheet (xlwings.Sheet): The Excel sheet object.
        header_name (str): The column header to search for.
    
    Returns:
        None
    """
    # Find the column index based on the header
    headers = sheet.range(f"{header_row}1").expand("right").value  # Read all headers in row 1
    if header_name not in headers:
        raise ValueError(f"Header '{header_name}' not found in the sheet.")
    col_index = headers.index(header_name) + 1  # Convert to Excel 1-based index
    col_letter = xw.utils.col_name(col_index)  # Convert to letter (e.g., B, C)
    # Apply percentage format with three-digit display (e.g., 100%, 045%, 008%)
    sheet.range(f"{col_letter}2:{col_letter}1048576").number_format = format


def format_dollar_column(sheet, header_name, format="$#,##0.00", header_row="A"):
    """
    Formats the column with the given header name as a dollar amount in an Excel sheet.

    Args:
        sheet (xlwings.Sheet): The Excel sheet object.
        header_name (str): The column header to search for.
        format (str, optional): The Excel number format for currency. Default is "$#,##0.00".
        header_row (str, optional): The row letter where headers are located. Default is "A".

    Returns:
        None
    """
    # Find the column index based on the header
    headers = sheet.range(f"{header_row}1").expand("right").value  # Read all headers in row 1
    if header_name not in headers:
        raise ValueError(f"Header '{header_name}' not found in the sheet.")
    
    col_index = headers.index(header_name) + 1  # Convert to Excel 1-based index
    col_letter = xw.utils.col_name(col_index)  # Convert to letter (e.g., B, C)
    
    # Apply dollar format (e.g., $1,234.56)
    sheet.range(f"{col_letter}2:{col_letter}1048576").number_format = format

def make_borders(sheet, linestyle=1, weight=2):
    used_range = sheet.used_range
    used_range.api.Borders.LineStyle = linestyle
    used_range.api.Borders.Weight = weight
