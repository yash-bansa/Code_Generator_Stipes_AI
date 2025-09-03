import pkgutil
import importlib
import inspect
import csv
import re
import sys

# --- regex patterns ---
_added_pat = re.compile(r'versionadded::\s*([0-9.]+)')
_changed_pat = re.compile(r'versionchanged::\s*([0-9.]+)\s*(.*)')
_depr_pat = re.compile(r'deprecated::\s*([0-9.]+)\s*(.*)', re.IGNORECASE)

def extract_version_info(doc: str):
    added = _added_pat.search(doc)
    changed = _changed_pat.search(doc)
    depr = _depr_pat.search(doc)
    replacement = ""

    if depr and depr.group(2):
        m = re.search(r"(?:use|replace with)\s+(.+?)(?:\.|$)", depr.group(2), re.IGNORECASE)
        if m:
            replacement = m.group(1).strip()

    return {
        'versionadded': added.group(1) if added else '',
        'versionchanged': changed.group(1) if changed else '',
        'versionchanged_note': changed.group(2).strip() if changed else '',
        'deprecated_in': depr.group(1) if depr else '',
        'deprecated_note': depr.group(2).strip() if depr else '',
        'replacement': replacement
    }

def is_deprecated(obj, doc):
    obj_name = getattr(obj, "__name__", "")
    return bool(_depr_pat.search(doc)) or (isinstance(obj_name, str) and obj_name.startswith('_deprecated'))

def get_api_metadata(module, qualname, obj, obj_type, pyspark_version):
    try:
        sig = str(inspect.signature(obj))
    except Exception:
        sig = ""
    doc = inspect.getdoc(obj) or ""
    doc_clean = doc.strip().replace("\r\n", "\n")
    verinfo = extract_version_info(doc)
    ret = getattr(obj, '__annotations__', {}).get("return", "")
    return {
        "pyspark_version": pyspark_version,
        "module": module,
        "qualname": qualname,
        "type": obj_type,
        "parameters": sig,
        "returns": str(ret),
        "deprecated": is_deprecated(obj, doc),
        "versionadded": verinfo['versionadded'],
        "versionchanged": verinfo['versionchanged'],
        "versionchanged_note": verinfo['versionchanged_note'],
        "deprecated_in": verinfo['deprecated_in'],
        "deprecated_note": verinfo['deprecated_note'],
        "replacement": verinfo['replacement'],
        "docstring": doc_clean
    }

def crawl_module(modname, pyspark_version):
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return []
    items = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        fullname = f"{modname}.{name}"
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            items.append(get_api_metadata(modname, fullname, obj, "function", pyspark_version))
        elif inspect.isclass(obj):
            items.append(get_api_metadata(modname, fullname, obj, "class", pyspark_version))
            for mname, meth in inspect.getmembers(obj):
                if mname.startswith("_"):
                    continue
                if inspect.isfunction(meth) or inspect.ismethod(meth) or isinstance(meth, staticmethod):
                    items.append(get_api_metadata(modname, f"{modname}.{name}.{mname}", meth, "method", pyspark_version))
        elif not callable(obj):  # constants, enums, etc.
            items.append(get_api_metadata(modname, fullname, obj, "attribute", pyspark_version))
    return items

def crawl_all():
    results = []
    base = "pyspark"
    pkg = importlib.import_module(base)
    pyspark_version = getattr(pkg, "__version__", "")
    for _, modname, _ in pkgutil.walk_packages(path=pkg.__path__, prefix=base + "."):
        print("Crawling", modname)
        results += crawl_module(modname, pyspark_version)
    return results

def write_csv(data, fname="new_pyspark_data_generation.csv"):
    keys = ["pyspark_version","module","qualname","type","parameters","returns","deprecated",
            "versionadded","versionchanged","versionchanged_note","deprecated_in","deprecated_note",
            "replacement","docstring"]
    with open(fname, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    all_data = crawl_all()
    write_csv(all_data)
    print("✅ Done. Saved to pyspark35_apis.csv")
