#TODO: resets downloaded bed files (deletes them)

import os

def delete_files_in_dir(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bed_dir = os.path.join(base_dir, 'data', 'bed')
    metadata_dir = os.path.join(base_dir, 'data', 'metadata')
    log_dir = os.path.join(base_dir, 'data', 'log')

    # delete files in ../data/bed/
    delete_files_in_dir(bed_dir)

    # delete files in ../data/metadata/
    delete_files_in_dir(metadata_dir)

    # delete files in ../data/log/
    delete_files_in_dir(log_dir)

if __name__ == "__main__":
    main()