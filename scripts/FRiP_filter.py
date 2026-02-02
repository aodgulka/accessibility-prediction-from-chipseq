# importing libraries
import pandas as pd
import os
import time
import threading
import requests
import json
import argparse

from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Take ENCODE tsv and download BED files with highest FRiP scores"
    )

    p.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input ENCODE TSV file"
    )

    p.add_argument(
        "-m", "--min_peaks",
        required=False,
        default=0,
        type=int,
        help="Minimum number of peaks to be considered a valid, high quality file"
    )

    p.add_argument(
        "-p", "--parallel",
        required=False,
        default=8,
        type=int,
        help="Number of parallel threads to use for pipeline"
    )

    return p.parse_args()

def split_into_chunks(lst, n):
    avg_len = len(lst)//n
    remainder = len(lst)%n
    chunks = []
    start = 0
    
    for i in range(n):
        # Each chunk gets an additional element if there's a remainder
        end = start + avg_len + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end

    return chunks

META_DIR = Path("../data/metadata")
META_DIR.mkdir(parents=True, exist_ok=True)

def download_metadata(accession_list, metadata_dict, lock):
    """
    Worker executed inside each thread.
    Adds a lock‑protected write to the shared dict so it’s thread‑safe.
    """
    headers = {"accept": "application/json"}

    for i, accession in enumerate(accession_list):
        if i % 10 == 0 and i:
            time.sleep(1)

        url = f"https://www.encodeproject.org/files/{accession}"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] {accession} failed: {e}")
            continue

        (META_DIR / f"{accession}.json").write_text(json.dumps(data))

        with lock:
            metadata_dict[accession] = data


def metadata_run(ENCODE_tsv, n):
    """
    Downloads metadata JSONS from samples specified in the input TSV file from the ENCODE api.
    """
    # Create the output directory "../data/metadata" . if already exists, delete all files in it
    os.makedirs("../data/metadata", exist_ok=True)
    # Delete all files in the directory
    for filename in os.listdir("../data/metadata"):
        file_path = os.path.join("../data/metadata", filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Read the TSV file
    ENCODE_df = pd.read_csv(ENCODE_tsv, sep='\t', skiprows=1)
    ENCODE_df = ENCODE_df[['Dataset', 'Accession', 'Target label']]

    #split list into n groups for parallel processing
    ENCODE_df = ENCODE_df['Accession'].tolist()
    bed_chunks = split_into_chunks(ENCODE_df, n)
    metadata_dict = {}
    threads = []

    # Create a lock for thread-safe access to the metadata_dict
    lock = threading.Lock()

    for chunk in bed_chunks:
        thread = threading.Thread(target=download_metadata, args=(chunk, metadata_dict, lock))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def read_metadata():
    # Create the output directory if it doesn't exist
    os.makedirs("../data/bed", exist_ok=True)
    bed_json_path = '../data/metadata/'

    BED_frip_dict = {}
    BED_dataset_dict = {}
    BED_target_dict = {}

    for json_file in os.listdir(bed_json_path):
        with open(bed_json_path + json_file) as f:
            data = json.load(f)
            if len(data['quality_metrics']) != 0:
                if 'frip' in data['quality_metrics'][0]:
                    frip = data['quality_metrics'][0]['frip']
                elif len(data['quality_metrics']) == 2 and 'frip' in data['quality_metrics'][1]:
                    frip = data['quality_metrics'][1]['frip']
                else:
                    frip = None
            else:
                frip = None

            dataset = data['dataset']
            target = data['target']['label']

            BED_frip_dict[json_file] = frip
            BED_dataset_dict[json_file] = dataset
            BED_target_dict[json_file] = target

    #save frip to df
    df_final = pd.DataFrame.from_dict(BED_frip_dict, orient='index', columns=['frip'])

    #add target column
    df_final['target'] = df_final.index
    df_final['target'] = df_final['target'].map(BED_target_dict)

    #add dataset column
    df_final['dataset'] = df_final.index
    df_final['dataset'] = df_final['dataset'].map(BED_dataset_dict)
    df_final['dataset'] = df_final['dataset'].str.split('/').str[2]

    #order by frip
    df_final = df_final.sort_values(by='frip', ascending=False)

    #keep first instance of target
    df_final = df_final.drop_duplicates(subset='target', keep='first')

    #parse index to get accession and store in column
    df_final['accession'] = df_final.index
    df_final['accession'] = df_final['accession'].str[:-5]

    #reset index
    df_final.reset_index(drop=True, inplace=True)

    #order columns as accession,target,dataset,frip
    df_final = df_final[['accession', 'target', 'dataset', 'frip']]
    
    return df_final

#download BED files and name them according to target format: "https://www.encodeproject.org/files/ENCFF636FWF/@@download/ENCFF636FWF.bed.gz"
def download_BED_file_helper(target, accession):
    os.system(f"wget -q -O ../data/bed/{target}.bed.gz 'https://www.encodeproject.org/files/{accession}/@@download/{accession}.bed.gz'")
    print(f"Downloaded {target}.bed")

def download_bed_file(accession_df, threads):
    """
    Make accession dictionary with target as key and accession as value. Download bed files from ENCODE.
    Logs each downloaded accession and target.
    """
    # keep first 3 columns
    accession_df = accession_df[['accession', 'target', 'dataset']]

    # make dict (key = target, value = accession) from pandas dataframe
    accession_dict = accession_df.set_index('target')['accession'].to_dict()

    active = []  # list of currently running Thread objects
    downloaded = []  # list to keep track of downloaded (target, accession)

    def download_and_log(target, accession):
        download_BED_file_helper(target, accession)
        downloaded.append((target, accession))

    for target, accession in accession_dict.items():
        # start a new thread for this download
        t = threading.Thread(target=download_and_log, args=(target, accession), daemon=True)
        t.start()
        active.append(t)

        # if we've hit our limit, wait for the oldest to finish
        if len(active) >= threads:
            active[0].join()
            active.pop(0)

    # wait for any remaining threads
    for t in active:
        t.join()

def drop_min_peaks(min_peaks):
    """
    Drops bed files with line counts less than min_peaks. --> gzipped bedfile
    """
    bed_dir = Path("../data/bed")
    for bed_file in bed_dir.glob("*.bed.gz"):
        with os.popen(f"zcat {bed_file} | wc -l") as f:
            line_count = int(f.read().strip())
            if line_count < min_peaks:
                os.remove(bed_file)
                print(f"Dropped {bed_file} with {line_count} peaks")

def logging(accession_df):
    """
    Log the downloaded BED files and their FRiP scores.
    """
    log_dir = Path("../data/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.txt"
    i = 1
    while log_file.exists():
        log_file = log_dir / f"log_{i}.txt"
        i += 1
    with log_file.open("w") as f:
        for _, row in accession_df.iterrows():
            f.write(f"{row['accession']}\t{row['target']}\t{row['dataset']}\t{row['frip']}\n")
    print(f"Log file created at {log_file}")


def main():
    args = parse_args()
    input_file = args.input
    min_peaks = int(args.min_peaks)
    parallel = args.parallel

    # Download metadata
    print("Downloading metadata...")
    metadata_run(input_file, parallel)
    print("Metadata downloaded.")

    # Read metadata and download BED files
    accession_df = read_metadata()
    print(f"Metadata read for tsv: {input_file}")
    
    # Download BED files
    download_bed_file(accession_df, parallel)

    # Drop BED files with line counts less than min_peaks
    if min_peaks > 0:
        drop_min_peaks(min_peaks)
        print(f"Dropped BED files with line counts less than {min_peaks}")

    logging(accession_df)

if __name__ == "__main__":
    main()