import pandas as pd
import docx2txt
from glob import glob
import re
import os

patient_reports = {}


def parse_report(patient_id, report_lines):
    global patient_reports
    mode = ''
    side = ''
    for line in report_lines:
        if line[:4] == 'acr ' or line[:11] == 'patient no.':
            continue
        if line == 'left breast:':
            side = 'L'
        elif line == 'right breast:':
            side = 'R'
        elif line == 'contrast enhanced spectral mammography revealed:':
            mode = 'CM'
        elif line == 'digitalized low dose soft tissue mammography revealed:':
            mode = 'DM'
        elif line == 'opinion:':
            mode = 'DM_OPINION'
        else:
            image_name = f"P{patient_id}_{side}_{mode}"
            if image_name in patient_reports.keys():
                patient_reports[image_name] += ' ' + line
            else:
                patient_reports[image_name] = line


report_csv = {"Image_name": [], "report": [], "opinion": []}

for filepath in glob('./reports/Medical reports/*.docx'):
    if os.path.basename(filepath)[0] == '~':
        continue
    print(filepath)
    report = docx2txt.process(filepath)
    patient_id = os.path.basename(filepath)[1:-5]
    report_lines = [line.strip().lower() for line in report.split('\n') if line.strip() != '']
    parse_report(patient_id, report_lines)

dataset_df = pd.read_excel('./reports/annotations.xlsx')

for index, row in dataset_df.iterrows():
    image_name = row['Image_name']
    report = patient_reports['_'.join(image_name.split('_')[:3])]
    opinion_key = ('_'.join(image_name.split('_')[:3]) + '_OPINION')
    if opinion_key in patient_reports:
        opinion = patient_reports[opinion_key]
    else:
        opinion = ''
    report_csv['Image_name'].append(image_name)
    report_csv['report'].append(report)
    report_csv['opinion'].append(opinion)

df = pd.DataFrame(report_csv)

df.to_csv(os.path.join("./reports", "reports.csv"), index=False)
