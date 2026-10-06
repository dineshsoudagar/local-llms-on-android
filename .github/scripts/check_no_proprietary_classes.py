#!/usr/bin/env python3
"""Fail if an APK contains classes from proprietary Google libraries.

Used on the F-Droid flavor, which must not ship Google ML Kit, Google Play Services or
Firebase. Scans the type descriptors in every classes*.dex file of each APK given.
"""
import re
import sys
import zipfile

FORBIDDEN = (
    b"Lcom/google/mlkit/",
    b"Lcom/google/android/gms/",
    b"Lcom/google/firebase/",
)
DEX_NAME = re.compile(r"^classes\d*\.dex$")


def scan(apk_path):
    hits = {}
    with zipfile.ZipFile(apk_path) as apk:
        dex_names = [name for name in apk.namelist() if DEX_NAME.match(name)]
        if not dex_names:
            raise SystemExit(f"{apk_path}: no classes*.dex found")
        for name in dex_names:
            data = apk.read(name)
            for prefix in FORBIDDEN:
                for match in re.finditer(re.escape(prefix) + rb"[A-Za-z0-9_/$]*;", data):
                    hits.setdefault(prefix.decode(), set()).add(f"{name}: {match.group().decode(errors='replace')}")
    return hits


def main(paths):
    if not paths:
        raise SystemExit("usage: check_no_proprietary_classes.py APK...")
    failed = False
    for path in paths:
        hits = scan(path)
        if hits:
            failed = True
            print(f"FAIL {path}")
            for prefix, entries in sorted(hits.items()):
                print(f"  {prefix} ({len(entries)} references)")
                for entry in sorted(entries)[:20]:
                    print(f"    {entry}")
        else:
            print(f"OK   {path}: no {', '.join(p.decode() for p in FORBIDDEN)} classes")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
