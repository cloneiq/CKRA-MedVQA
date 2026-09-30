import json

from rdflib import Graph, URIRef, Literal, RDF, RDFS

g = Graph()
g.parse("RadLex.owl")

# Read the mapping dictionary from RID to English labels
with open("rid_to_label.json", "r", encoding="utf-8") as f:
    rid_to_label = json.load(f)

# Read the original TXT file
with open("RIDRadLex.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Create a new list to store the processed triples
processed_triples = []

for line in lines:
    # Remove the newline character at the end of the line
    line = line.strip()

    # Split the triple using "%"
    parts = line.split("%")

    # Extract the RID, predicate, and object
    if len(parts) != 3:
        continue  # Skip this line if the format is incorrect

    subj, pred, obj = parts[0], parts[1], parts[2]

    subj = subj.strip("<>")
    pred = pred.strip("<>")
    obj = obj.strip("<>")

    if subj == obj:
        continue  # Skip this line if the subject and object are identical

    # Replace the RID with the corresponding English label
    rid = subj
    if rid in rid_to_label:
        subj = f'{rid_to_label[rid]}'  # Wrap the English label in quotes
    else:
        continue

    if "RID" in obj:
        rid = obj
        if rid in rid_to_label:
            obj = f'{rid_to_label[rid]}'
        else:
            continue

    # Check whether the predicate is <Preferred_name>
    if "<Preferred_name>" in pred:
        continue  # Skip this triple


    # Add the processed triple to the list
    processed_triples.append(f"{subj}%{pred}%{obj}")

# Write the processed triples to a new TXT file
with open("RadLex.txt", "w", encoding="utf-8") as f:
    for triple in processed_triples:
        f.write(triple + "\n")

print("Finished.")
