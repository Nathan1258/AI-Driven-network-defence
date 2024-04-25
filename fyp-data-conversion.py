import csv
import re

input_log_file = './conn.log.labeled'
output_csv_file = './output.csv'

field_names = []
data_rows = []

field_headings = {
    'ts': 'Timestamp',
    'uid': 'Unique_ID',
    'id.orig_h': 'Origin_IP',
    'id.orig_p': 'Origin_Port',
    'id.resp_h': 'Destination_IP',
    'id.resp_p': 'Destination_Port',
    'proto': 'Protocol',
    'service': 'Service',
    'duration': 'Duration',
    'orig_bytes': 'Origin_Bytes',
    'resp_bytes': 'Response_Bytes',
    'conn_state': 'Connection_State',
    'local_orig': 'Local_Origin',
    'local_resp': 'Local_Response',
    'missed_bytes': 'Missed_Bytes',
    'history': 'History',
    'orig_pkts': 'Origin_Packets',
    'orig_ip_bytes': 'Origin_IP_Bytes',
    'resp_pkts': 'Response_Packets',
    'resp_ip_bytes': 'Response_IP_Bytes',
    'tunnel_parents': 'Tunnel_Parents',
    'label': 'Label',
    'detailed-label': 'Detailed_Label'
}

with open(input_log_file, 'r') as log_file:
    for line in log_file:
        if line.startswith('#fields'):
            field_names_str = re.search(r'#fields\s+(.*)', line).group(1)
            field_names = re.split(r'\s+', field_names_str.strip())
        elif not line.startswith('#') and field_names:
            fields = re.split(r'\s+', line.strip())
            data_dict = {field_headings[field_names[i]]: fields[i] if i < len(fields) else '' for i in range(len(field_names))}
            data_rows.append(data_dict)

with open(output_csv_file, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=[field_headings[field_name] for field_name in field_names])
    writer.writeheader()
    writer.writerows(data_rows)

print('Conversion to CSV complete.')
