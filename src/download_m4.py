"""Download the M4 competition data used by `src/evaluate_m4.py`.

Source: github.com/Mcompetitions/M4-methods, folder `Dataset/`. Written to
`<data_dir>/{Train,Test}/<Frequency>-{train,test}.csv` and `<data_dir>/M4-info.csv` (default `datasets/m4`).
Existing files are skipped (use `--force` to download again); every file is written to a temporary file first
and renamed when complete, so an interrupted download leaves no partial CSV behind.

Run from the repository root:
    python -m src.download_m4                       # all frequencies (about 275 MB)
    python -m src.download_m4 --frequencies Yearly  # only the yearly series (about 31 MB)
"""
import argparse
import os
import shutil
import urllib.request

BASE_URL = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset'
FREQUENCIES = ('Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly')


def files_to_download(frequencies):
    """Relative paths (also relative to `BASE_URL`) of the files needed for `frequencies`."""
    paths = ['M4-info.csv']
    for frequency in frequencies:
        paths += [f'Train/{frequency}-train.csv', f'Test/{frequency}-test.csv']
    return paths


def download(url, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.part'
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, 'wb') as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download the M4 competition data (CSV files).')
    parser.add_argument('--data_dir', default='datasets/m4', help='Target folder (as `evaluate_m4.py --data_dir`).')
    parser.add_argument('--frequencies', nargs='+', choices=FREQUENCIES, default=list(FREQUENCIES))
    parser.add_argument('--force', action='store_true', help='Download files that already exist again.')
    args = parser.parse_args(argv)

    for rel in files_to_download(args.frequencies):
        path = os.path.join(args.data_dir, rel)
        if os.path.exists(path) and not args.force:
            print(f'exists, skipped: {path}')
            continue
        print(f'downloading {BASE_URL}/{rel} ...', end=' ', flush=True)
        download(f'{BASE_URL}/{rel}', path)
        print(f'{os.path.getsize(path) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
