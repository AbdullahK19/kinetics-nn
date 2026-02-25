"""
Excel driver module for interacting with the PFR-CSTR cascade workbook via xlwings.

This module provides functions to:
- Open the Excel workbook
- Set input parameters in the 'preparation' sheet
- Trigger recalculation
- Read concentration-time profiles from PFR and CSTR sheets
"""

import xlwings as xw
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import time


def open_workbook(path: str, visible: bool = False) -> Tuple[xw.App, xw.Book]:
    """
    Open the Excel workbook with xlwings.

    Args:
        path: Path to the .xlsm workbook
        visible: Whether to make Excel visible (useful for debugging)

    Returns:
        Tuple of (app, workbook)
    """
    app = xw.App(visible=visible)
    app.display_alerts = False
    app.screen_updating = False

    wb = app.books.open(path)
    return app, wb


def set_inputs(wb: xw.Book, params: Dict[str, float]) -> None:
    """
    Set input parameters in the 'preparation' sheet.

    Expected params keys:
        - nA0: n(dot,A,0) -> I10
        - nB0: n(dot,B,0) -> I11
        - nI0: n(dot,I,0) -> I14
        - k1: k1 -> I15
        - k2: k2 -> I16
        - n_cstr: n(CSTR) -> I3 (optional, defaults to existing value)
        - V_tot: V(tot) -> I6 (optional, defaults to existing value)
        - A_pfr: A(PFR,CS) -> I7 (optional, defaults to existing value)
        - p: pressure -> I4 (optional, defaults to 100000 Pa)

    Args:
        wb: xlwings workbook object
        params: Dictionary of parameter names and values
    """
    prep_sheet = wb.sheets['preparation']

    # Required parameters
    prep_sheet.range('I10').value = params['nA0']
    prep_sheet.range('I11').value = params['nB0']
    prep_sheet.range('I14').value = params['nI0']
    prep_sheet.range('I15').value = params['k1']
    prep_sheet.range('I16').value = params['k2']

    # Optional parameters (only set if provided)
    if 'n_cstr' in params:
        prep_sheet.range('I3').value = params['n_cstr']
    if 'V_tot' in params:
        prep_sheet.range('I6').value = params['V_tot']
    if 'A_pfr' in params:
        prep_sheet.range('I7').value = params['A_pfr']
    if 'p' in params:
        prep_sheet.range('I4').value = params['p']


def recalc(wb: xw.Book, wait_time: float = 0.5) -> None:
    """
    Trigger full recalculation of the workbook.

    Args:
        wb: xlwings workbook object
        wait_time: Time to wait after recalculation (seconds)
    """
    # Force full recalculation (cross-platform compatible)
    wb.app.calculate()

    # Wait for calculations to complete
    time.sleep(wait_time)


def find_last_row(sheet: xw.Sheet, start_row: int = 10, col: str = 'A') -> int:
    """
    Find the last row with data in a given column.

    Args:
        sheet: xlwings sheet object
        start_row: Row number to start searching from
        col: Column letter to check

    Returns:
        Last row number with data
    """
    row = start_row
    max_search = 10000  # Safety limit

    while row < start_row + max_search:
        cell_value = sheet.range(f'{col}{row}').value
        if cell_value is None or cell_value == '':
            return row - 1
        row += 1

    return row - 1


def read_pfr_profile(wb: xw.Book) -> Dict[str, np.ndarray]:
    """
    Read concentration profile from the PFR sheet.

    The PFR sheet has:
    - Header row at row 9
    - Data starts at row 10
    - Columns: z / m, V / m3, n(dot,A,x), n(dot,B,x), n(dot,C,x), n(dot,D,x), n(dot,total), ...

    Returns:
        Dictionary with keys: 'z', 'V', 'A', 'B', 'C', 'D', 'n_total' (numpy arrays)
    """
    pfr_sheet = wb.sheets['PFR']

    # Find last row with data
    last_row = find_last_row(pfr_sheet, start_row=10, col='A')

    if last_row < 10:
        raise ValueError("No data found in PFR sheet")

    # Read data range
    # Assuming columns: A=z, B=V, C=nA, D=nB, E=nC, F=nD, G=nI
    data_range = pfr_sheet.range(f'A10:G{last_row}').value

    # Convert to numpy array
    data = np.array(data_range)

    # Extract columns
    result = {
        'z': data[:, 0],      # z / m
        'V': data[:, 1],      # V / m3
        'A': data[:, 2],      # n(dot,A,x)
        'B': data[:, 3],      # n(dot,B,x)
        'C': data[:, 4],      # n(dot,C,x)
        'D': data[:, 5],      # n(dot,D,x)
        'n_total': data[:, 6],  # n(dot,total,x)
    }

    return result


def read_cstr_profile(wb: xw.Book) -> Dict[str, np.ndarray]:
    """
    Read concentration profile from the CSTR sheet.

    The CSTR sheet has:
    - Header row at row 9
    - Data starts at row 10
    - Columns: n, V, n(dot,A,e), n(dot,B,e), n(dot,C,e), n(dot,D,e), n(dot,total), ...

    Returns:
        Dictionary with keys: 'n', 'V', 'A', 'B', 'C', 'D', 'n_total' (numpy arrays)
    """
    cstr_sheet = wb.sheets['CSTR']

    # Find last row with data
    last_row = find_last_row(cstr_sheet, start_row=10, col='A')

    if last_row < 10:
        raise ValueError("No data found in CSTR sheet")

    # Read data range
    # Assuming columns: A=n (stage), B=V, C=nA, D=nB, E=nC, F=nD, G=nI
    data_range = cstr_sheet.range(f'A10:G{last_row}').value

    # Convert to numpy array
    data = np.array(data_range)

    # Extract columns
    result = {
        'n': data[:, 0],      # stage number
        'V': data[:, 1],      # V
        'A': data[:, 2],      # n(dot,A,e)
        'B': data[:, 3],      # n(dot,B,e)
        'C': data[:, 4],      # n(dot,C,e)
        'D': data[:, 5],      # n(dot,D,e)
        'n_total': data[:, 6],  # n(dot,total,e)
    }

    return result


def close_workbook(app: xw.App, wb: xw.Book, save: bool = False) -> None:
    """
    Close the workbook and quit Excel.

    Args:
        app: xlwings app object
        wb: xlwings workbook object
        save: Whether to save changes before closing
    """
    wb.close()
    app.quit()


def test_excel_driver(workbook_path: str) -> None:
    """
    Test the excel driver with sample parameters.

    Args:
        workbook_path: Path to the Excel workbook
    """
    print("Testing Excel driver...")

    # Open workbook
    app, wb = open_workbook(workbook_path, visible=True)

    try:
        # Set test parameters
        test_params = {
            'nA0': 0.2,
            'nB0': 0.15,
            'nI0': 0.05,
            'k1': 1e-10,
            'k2': 0.0,
            'p': 100000
        }

        print(f"Setting parameters: {test_params}")
        set_inputs(wb, test_params)

        # Recalculate
        print("Recalculating...")
        recalc(wb, wait_time=1.0)

        # Read PFR profile
        print("Reading PFR profile...")
        pfr_data = read_pfr_profile(wb)
        print(f"PFR data points: {len(pfr_data['z'])}")
        print(f"PFR z range: {pfr_data['z'][0]:.4f} to {pfr_data['z'][-1]:.4f}")
        print(f"PFR concentrations at end: A={pfr_data['A'][-1]:.6f}, B={pfr_data['B'][-1]:.6f}, "
              f"C={pfr_data['C'][-1]:.6f}, D={pfr_data['D'][-1]:.6f}")

        # Read CSTR profile
        print("\nReading CSTR profile...")
        cstr_data = read_cstr_profile(wb)
        print(f"CSTR data points: {len(cstr_data['n'])}")
        print(f"CSTR stages: {cstr_data['n'][0]:.0f} to {cstr_data['n'][-1]:.0f}")
        print(f"CSTR concentrations at end: A={cstr_data['A'][-1]:.6f}, B={cstr_data['B'][-1]:.6f}, "
              f"C={cstr_data['C'][-1]:.6f}, D={cstr_data['D'][-1]:.6f}")

        print("\nTest completed successfully!")

    finally:
        # Close workbook
        close_workbook(app, wb, save=False)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        workbook_path = sys.argv[1]
        test_excel_driver(workbook_path)
    else:
        print("Usage: python excel_driver.py <path_to_workbook.xlsm>")
        print("\nExample:")
        print("  python excel_driver.py /mnt/data/PFR-CSTR-cascade_Calculation-macro.xlsm")
