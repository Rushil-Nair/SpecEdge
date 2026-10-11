"""Flag call tasks whose expected arguments use a value the question never gives.

    python data/flag_unstated_values.py <candidate tasks .jsonl>

A test task is only fair if the expected call can be worked out from the question and the
tool descriptions. xLAM has tasks where it cannot: an id, a URL or coordinates that appear
only in the expected answer. This script does the mechanical half of finding them: it lists
every expected argument value that does not appear in the question (as text, as a number, or
as a percentage) and is not the tool's default. It over-flags on purpose. Values that are
implied ("July" for month 7, "Germany" for "DE", "5 feet 9" for 69 inches) are flagged too,
so every flagged task is then read by a person and the ones to drop are written, with the
reason, to a file such as data/toolset/v1/dropped_unstated_values.json.
"""
import json
import re
import sys

tasks = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
def norm(s): return re.sub(r"[^a-z0-9.]+"," ",str(s).lower()).strip()
def found(v,q,qn):
    if isinstance(v,bool) or v is None: return True
    if isinstance(v,(int,float)):
        forms={str(v)}
        if float(v)==int(v): forms|={str(int(v)), f"{int(v):,}"}
        forms.add(("%g"%(v*100)))  # percent
        return any(re.search(r"(?<![\d.])"+re.escape(f)+r"(?![\d])",q) for f in forms)
    if isinstance(v,str):
        if v.strip()=="" : return True
        try:
            float(v); return found(float(v),q,qn) or v in q
        except ValueError: pass
        vn=norm(v)
        if vn in qn: return True
        toks=[t for t in vn.split() if len(t)>1]
        return bool(toks) and all(t in qn for t in toks)
    if isinstance(v,list): return all(found(x,q,qn) for x in v)
    if isinstance(v,dict): return all(found(x,q,qn) for x in v.values())
    return True
out=[]
for t in tasks:
    if t["expect"]["kind"]!="calls": continue
    q=t["messages"][0]["content"]; qn=norm(q); miss=[]
    for c in t["expect"]["calls"]:
        tool=next(x for x in t["tools"] if x["function"]["name"]==c["name"])
        props=tool["function"]["parameters"].get("properties",{})
        arguments = c.get("arguments")
        if arguments is None:        # BFCL form: each argument lists its acceptable values; "" means optional
            arguments = {k: (alts if isinstance(alts, list) else [alts]) for k, alts in c["alternatives"].items()}
            arguments = {k: alts for k, alts in arguments.items() if "" not in alts}
        else:
            arguments = {k: [v] for k, v in arguments.items()}
        for k,alts in arguments.items():
            d=props.get(k,{}).get("default")
            if d is not None and any(str(d).lower()==str(v).lower() for v in alts): continue
            if not any(found(v,q.lower(),qn) for v in alts): miss.append((c["name"],k,alts if len(alts)>1 else alts[0],d,props.get(k,{}).get("description","")[:90]))
    if miss: out.append((t["id"],q,miss))
print(len(out),"call tasks flagged")
for i,q,m in out:
    print("--",i,"|",q[:300])
    for x in m: print("     ",x[0],"|",x[1],"=",json.dumps(x[2],ensure_ascii=False)[:90],"| default",x[3],"|",x[4])
