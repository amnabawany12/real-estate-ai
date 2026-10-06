import os, json
chunks=[]
for file in os.listdir("knowledge_base"):
    with open(os.path.join("knowledge_base",file), encoding='utf-8') as f:
        text=f.read()
        for i in range(0,len(text),500):
            t=text[i:i+500].strip()
            if len(t)>30:
                chunks.append({"text":t,"source":file})
with open("chunks.json","w", encoding="utf-8") as out:
    json.dump(chunks,out, indent=2)
print(f"DONE! {len(chunks)} chunks saved")