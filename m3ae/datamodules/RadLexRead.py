import json

from rdflib import Graph, URIRef, Literal, RDF, RDFS

g = Graph()
g.parse("RadLex.owl")

# 读取RID到英文标签的映射字典
with open("rid_to_label.json", "r", encoding="utf-8") as f:
    rid_to_label = json.load(f)

# 读取原始TXT文件
with open("RIDRadLex.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 创建一个新的列表来存储处理后的三元组
processed_triples = []

for line in lines:
    # 清理行末尾的换行符
    line = line.strip()

    # 将三元组用“%”分割
    parts = line.split("%")

    # 提取RID、谓词和对象
    if len(parts) != 3:
        continue  # 如果格式不正确，跳过此行

    subj, pred, obj = parts[0], parts[1], parts[2]

    subj = subj.strip("<>")
    pred = pred.strip("<>")
    obj = obj.strip("<>")

    if subj == obj:
        continue  # 如果主语和宾语相同，跳过此行

    # 替换RID为对应的英文标签
    rid = subj
    if rid in rid_to_label:
        subj = f'{rid_to_label[rid]}'  # 使用引号包裹英文标签
    else:
        continue

    if "RID" in obj:
        rid = obj
        if rid in rid_to_label:
            obj = f'{rid_to_label[rid]}'
        else:
            continue

    # 检查谓词是否为<Preferred_name>
    if "<Preferred_name>" in pred:
        continue  # 跳过这个三元组


    # 添加处理后的三元组到列表中
    processed_triples.append(f"{subj}%{pred}%{obj}")

# 将处理后的三元组写入新的TXT文件
with open("RadLex.txt", "w", encoding="utf-8") as f:
    for triple in processed_triples:
        f.write(triple + "\n")

print("处理完成，已保存到文件中。")