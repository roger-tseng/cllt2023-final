from glob import glob
import opencc

t2s = opencc.OpenCC('t2s.json')
s2t = opencc.OpenCC('s2t.json')

def get_files(year):
    if year not in ['2020', '2021', '2022', '2023']:
        year = '**'
    files = glob(f'data/HatePolitics/{year}/*.xml', recursive=True)
    return files

def trad_to_simp(text):
    return t2s.convert(text)

def simp_to_trad(text):
    return s2t.convert(text)

if __name__ == '__main__':
    print(len(get_files('0')))