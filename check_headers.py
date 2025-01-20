import os

def check_headers_for_labels(folder):
    records = [f for f in os.listdir(folder) if f.endswith('.hea')]
    for record in sorted(records)[-10:]:  # Check last 10 records
        with open(os.path.join(folder, record), 'r') as f:
            header = f.read()
            print(f"Header for {record}:")
            print(header)
            if '# Labels:' not in header:
                print(f"Missing labels for {record}")
            else:
                print(f"Labels present for {record}")
            print('-' * 50)

# Adjust the folder path as needed
header_folder = "/scratch/project_2010663/ECG_digitalization/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/00000"
check_headers_for_labels(header_folder)
