import pandas as pd
import csv
df = pd.read_csv("./dataset/labelled_newscatcher_dataset.csv", encoding='utf-8', sep=';')

labels = set(df['topic'])
contents = df['title']
count = {}
cal = {}
for p in df['topic']:
    cal[p] = 0
    try:
        count[p] += 1
    except KeyError:
        count[p] = 1
print(count)

train, val, test = [], [], []
for i, label in enumerate(df['category']):
    if cal[label] < count[label] * 0.7:
        train.append({'label': label, 'content': contents[i]})
    elif cal[label] < count[label] * 0.85:
        val.append({'label': label, 'content': contents[i]})
    else:
        test.append({'label': label, 'content': contents[i]})
    cal[label] += 1

with open('./dataset/train.csv', 'a', newline='', encoding='utf-8') as f:
    xieru = csv.DictWriter(f, ['label','content'],delimiter=';')
    xieru.writerows(train)  # writerows方法是一下子写入多行内容
with open('./dataset/val.csv', 'a', newline='', encoding='utf-8') as f:
    xieru = csv.DictWriter(f, ['label','content'],delimiter=';')
    xieru.writerows(val)  # writerows方法是一下子写入多行内容
with open('./dataset/test.csv', 'a', newline='', encoding='utf-8') as f:
    xieru = csv.DictWriter(f, ['label','content'],delimiter=';')
    xieru.writerows(test)  # writerows方法是一下子写入多行内容
