import xml.etree.ElementTree as ET

# Parse the XML data
tree = ET.parse('/content/data/HatePolitics/2021/20210103_0511_M.1609650710.A.BB8.xml')
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
comments = text.findall('comment')
comments_pairs = [([(word.get('type'), word.text) for word in c.findall('s/w')], c.get('c_type')) for c in comments]
sentences_pairs = [[(word.get('type'), word.text) for word in sent.findall('w')] for sent in sentences]

print(sentences_pairs)
print(comments_pairs)
