from datetime import datetime
import calendar

def movecol(df, cols_to_move, ref_col, place='After'):
    """Move columns inside a DataFrame"""
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    seg1 = [pair for pair in seg1 if pair not in seg2]
    seg3 = [pair for pair in cols if pair not in seg1 + seg2]
    return df[seg1 + seg2 + seg3]

def get_csv_list(root):
    """Transforms a .xlm to a .csv"""
    csv_list = []
    for child in root:
        for section in child:
            for row in section:
                row_list = []
                csv_list.append(row_list)
                for cell in row:
                    for data in cell:
                        row_list.append(data.text)
    return csv_list

def add_months(date, months):
    """Add months on each step based on an initial date and a brick size in months"""
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y.%m.%d')
    month = date.month - 1 + months
    year = date.year + month
    month = month % 12 + 1
    day = min(date.day, calendar.monthrange(year, month)[1])

    return datetime(year, month, day)

