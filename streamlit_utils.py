
import os
import base64
from io import BytesIO
import pandas as pd 

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def to_excel_multi(dfs, sheetnames, use_index):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, sheetname in zip(dfs, sheetnames):
            df.to_excel(writer, sheet_name=sheetname, index=use_index)
        writer.save()
    processed_data = output.getvalue()
    return processed_data

def to_excel_one(df, use_index):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=use_index)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def df2excel(dfs, sheetnames, filename, xltype="single", use_index=False):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    if xltype=="single":
        val = to_excel_one(dfs, use_index)
    elif xltype == "multi":
        val = to_excel_multi(dfs, sheetnames, use_index)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.xlsx">Download excel file</a>' # decode b'abc' => abc
