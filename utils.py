from glob import glob

def get_files(year):
    if year not in ['2020', '2021', '2022', '2023']:
        year = '**'
    files = glob(f'data/HatePolitics/{year}/*.xml', recursive=True)
    return files

if __name__ == '__main__':
    print(len(get_files('0')))