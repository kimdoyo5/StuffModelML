import csv
import pprint

def process_dataset(file_path, skip_header=True, skip_first_column=False):
    dataset = []

    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        if skip_header:
            next(file)

        reader = csv.reader(file)
        headers = next(reader)
        if skip_first_column:
            headers = headers[1:]

        headers = [header.strip().strip('"').strip("'") for header in headers]

        for row in reader:
            if skip_first_column:
                row = row[1:]
            processed_row = {
                headers[i]: (str(value).strip() if str(value).strip() else None)
                for i, value in enumerate(row)
            }
            dataset.append(processed_row)

    return dataset

def merge_datasets(dataset1, dataset2):
    ra9_mapping = {row['mlbid']: row['RA9'] for row in dataset2}

    for row in dataset1:
        pitcher_id = row.get('pitcher', None)
        row['RA9'] = ra9_mapping.get(pitcher_id, None)

    return dataset1

def filter_by_year(dataset, years):
    year_set = set(str(year) for year in years)
    filtered_dataset = [
        row for row in dataset
        if 'game_date' in row and row['game_date'] and row['game_date'][:4] in year_set
    ]
    return filtered_dataset

def format_print(data, num_rows=1):
    pp = pprint.PrettyPrinter(indent=2, width=120, sort_dicts=False)
    for i, row in enumerate(data[:num_rows]):
        print(f"Row {i+1}:")
        pp.pprint(row)
        print()

def save_dataset(dataset, file_name):
    if not dataset:
        print("The dataset is empty. Nothing to save.")
        return

    headers = dataset[0].keys()

    try:
        with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(dataset)
        print(f"Dataset successfully saved to {file_name}.")
    except Exception as e:
        print(f"An error occurred while saving the dataset: {e}")


if __name__ == '__main__':
    original_file_path = "savant_data_2020-2024_updated.csv"
    ra9_2020 = "RA9 data/RA9 2020.csv"
    ra9_2021 = "RA9 data/RA9 2021.csv"
    ra9_2022 = "RA9 data/RA9 2022.csv"
    ra9_2023 = "RA9 data/RA9 2023.csv"
    ra9_2024 = "RA9 data/RA9 2024.csv"

    # Process all datasets
    original_dataset = process_dataset(original_file_path, True, True)
    ra9_2020_dataset = process_dataset(ra9_2020, False, False)
    ra9_2021_dataset = process_dataset(ra9_2021, False, False)
    ra9_2022_dataset = process_dataset(ra9_2022, False, False)
    ra9_2023_dataset = process_dataset(ra9_2023, False, False)
    ra9_2024_dataset = process_dataset(ra9_2024, False, False)
    
    # Filter by year
    train_set = filter_by_year(original_dataset, [2020, 2021, 2022])
    val_set = filter_by_year(original_dataset, [2023])
    test_set = filter_by_year(original_dataset, [2024])

    # Merge RA9 by year
    train_set = merge_datasets(train_set, ra9_2020_dataset)
    train_set = merge_datasets(train_set, ra9_2021_dataset)
    train_set = merge_datasets(train_set, ra9_2022_dataset)
    val_set = merge_datasets(val_set, ra9_2023_dataset)
    test_set = merge_datasets(test_set, ra9_2024_dataset)

    # Save datasets
    save_dataset(train_set, "data/train_set.csv")
    save_dataset(val_set, "data/val_set.csv")
    save_dataset(test_set, "data/test_set.csv")
