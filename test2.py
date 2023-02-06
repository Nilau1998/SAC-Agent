import os
import csv
import glob

if __name__ == '__main__':
    def read_last_line(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1]
        return last_line

    def append_to_csv(data, filename):
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    filenames = glob.glob(
        'experdadaiments_backup/setting_3/*/episodes/info.csv')
    new_filename = 'all_data.csv'

    for filename in filenames:
        last_line = read_last_line(filename)
        data = last_line.split(',')
        append_to_csv(data, new_filename)
