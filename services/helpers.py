def movecol(df, cols_to_move, ref_col, place='After'):
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

