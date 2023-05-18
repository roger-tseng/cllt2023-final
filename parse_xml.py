import xml.etree.ElementTree as ET

# Parse the XML data
tree = ET.parse('/content/politics/2020/20200101_2018_M.1577909917.A.8FE.xml')
root = tree.getroot()

# Extract metadata
metadata = root.find('./teiHeader')
metadata_dict = {}
for item in metadata.findall('metadata'):
    name = item.get('name')
    value = item.text
    metadata_dict[name] = value

# Extract text content
text = root.find('./text')
body_author = text.find('./body').get('author')
title_author = text.find('./title').get('author')
sentences = text.findall('body/s')
sentences_and_pairs = [[(word.get('type'), word.text) for word in sent.findall('w')] for sent in sentences]
# "".join([element.text for element in text.findall('body/s/w')]) # text.find('./title/s').text.strip()

# Print the extracted information
# print('Metadata:')
# for name, value in metadata_dict.items():
#     print(f'{name}: {value}')

# print('\nText Content:')
# print(f'Body Author: {body_author}')
# print(f'Title Author: {title_author}')
# print(f'Sentence: {sentence}')
print(sentences_and_pairs)
