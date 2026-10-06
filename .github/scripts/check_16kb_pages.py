#!/usr/bin/env python3
"""Check that 64-bit native libraries in an APK support 16 KB memory pages.

Google Play requires this for apps targeting Android 15 or newer. A library passes when
every PT_LOAD segment is aligned to at least 16 KB and, if stored uncompressed, its data
starts at a 16 KB boundary inside the APK.
"""
import struct
import sys
import zipfile

PAGE = 16 * 1024
ABIS = ("arm64-v8a", "x86_64")


def load_alignments(data):
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    endian = "<" if data[5] == 1 else ">"
    phoff = struct.unpack_from(endian + "Q", data, 0x20)[0]
    phentsize, phnum = struct.unpack_from(endian + "HH", data, 0x36)
    aligns = []
    for index in range(phnum):
        base = phoff + index * phentsize
        p_type = struct.unpack_from(endian + "I", data, base)[0]
        if p_type == 1:  # PT_LOAD
            aligns.append(struct.unpack_from(endian + "Q", data, base + 0x30)[0])
    return aligns


def data_offset(apk_path, info):
    with open(apk_path, "rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    return info.header_offset + 30 + name_len + extra_len


def main(apk_path):
    failures = []
    checked = 0
    with zipfile.ZipFile(apk_path) as apk:
        for info in apk.infolist():
            parts = info.filename.split("/")
            if len(parts) != 3 or parts[0] != "lib" or parts[1] not in ABIS or not parts[2].endswith(".so"):
                continue
            checked += 1
            aligns = load_alignments(apk.read(info))
            if aligns is None:
                failures.append(f"{info.filename}: not a 64-bit ELF file")
                continue
            too_small = [a for a in aligns if a < PAGE]
            if too_small:
                failures.append(f"{info.filename}: LOAD segment alignment {min(too_small)} < {PAGE}")
            if info.compress_type == zipfile.ZIP_STORED and data_offset(apk_path, info) % PAGE:
                failures.append(f"{info.filename}: stored uncompressed but not 16 KB aligned in the APK")
    print(f"Checked {checked} native libraries in {apk_path}")
    for failure in failures:
        print(f"::warning title=16 KB page size::{failure}")
    if not failures:
        print("All 64-bit native libraries support 16 KB pages.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
