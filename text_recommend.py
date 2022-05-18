import pandas as pd
import spacy
from numpy import dot
from numpy.linalg import norm
import numpy as np

np.seterr(invalid='ignore')
csv_data = pd.read_csv("ktu_didziuju_ld2_final.csv", sep = ';', dtype = {'qwikidata_id': str, 'instance_of': str, 'city_name': str, 'pageview_count': float, 'registered_contributors_count': float, 'anonymous_contributors_count': float, 'num_wikipedia_lang_pages': float, 'is_shop': int, 'is_tourism': int, 'is_leisure': int, 'is_other': int, 'has_image': int, 'description': str, 'image_files': str})
csv_data = csv_data[csv_data['description'].notna()]
csv_data.drop_duplicates('qwikidata_id', inplace=True)
csv_data.reset_index(inplace=True)
csv_data = csv_data
all_descriptions = csv_data['description']

nlp = spacy.load('/Users/gintas/Documents/SchoolProjects/didziuju/2/venv/lib/python3.7/site-packages/en_core_web_lg/en_core_web_lg-3.3.0')

global_entities = []
# global_labels_counts = {}
count = 0
for description in all_descriptions:
    print(f"{count}/{len(all_descriptions)}")
    text = description

    doc = nlp(text)
    for ent in doc.ents:
        global_entities.append(str(ent))
        # if ent.label_ in global_labels_counts:
        #     global_labels_counts[ent.label_] += 1
        # else:
        #     global_labels_counts[ent.label_] = 1
    count += 1

print(len(global_entities))

vectors = []
row_entities = []
count = 0
for index, row in csv_data.iterrows():
    print(f"{index}/{len(csv_data)}")
    row_description = row["description"]
    vector = [0] * len(global_entities)
    local_entities = []
    try:
        doc = nlp(row_description)
        for ent in doc.ents:
            local_entity = str(ent)
            local_entities.append(local_entity)
            index = global_entities.index(local_entity)
            vector[index] = 1
    except:
        None
    vectors.append(vector)
    local_entities = sorted(local_entities)
    row_entities.append(local_entities)
csv_data["description_vector"] = vectors
csv_data["entities"] = row_entities

# search
search_qwikidata_id = 'Q23929396'
search_row = csv_data[csv_data['qwikidata_id'] == search_qwikidata_id]
search_row_vector = search_row["description_vector"].tolist()[0]

def calc_cos_two_vectors(vec1, vec2):
    bottom = (norm(vec1) * norm(vec2))
    if bottom == 0:
        return 0
    cos_sim = dot(vec1, vec2) / bottom
    return cos_sim

csv_data['cos_angle'] = csv_data.apply(lambda row: calc_cos_two_vectors(search_row_vector, row['description_vector']), axis=1)
minimal_csv_data = csv_data[['qwikidata_id', 'description', 'cos_angle', 'entities']]
minimal_csv_data = minimal_csv_data[~pd.isna(minimal_csv_data['cos_angle'])]
minimal_csv_data = minimal_csv_data.sort_values('cos_angle', ascending = False)
minimal_csv_data = minimal_csv_data[1:10]

print(f"Searched for objects similar to this:")
print(f"qwikidata_id: {search_row['qwikidata_id'].values[0]}")
print(f"description: {search_row['description'].values[0]}")
print(f"entities: {','.join(search_row['entities'].values[0])}")
print(f"\n\nFound these similar results:")
for i in range(0, len(minimal_csv_data)):
    print(f"{i + 1}. qwikidata_id: {minimal_csv_data['qwikidata_id'].values[i]}")
    print(f"{i + 1}. description: {minimal_csv_data['description'].values[i]}")
    print(f"{i + 1}. cos (similarity): {minimal_csv_data['cos_angle'].values[i]}")
    print(f"{i + 1}. entities: {','.join(minimal_csv_data['entities'].values[i])}")
    print(f"========================================================================")