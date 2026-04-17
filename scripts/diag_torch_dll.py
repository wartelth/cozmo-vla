import os, sys, glob, ctypes, pefile

TORCH_LIB = r"C:\Users\thiba\miniconda3\envs\dl-gpu\Lib\site-packages\torch\lib"
ENV_ROOT = r"C:\Users\thiba\miniconda3\envs\dl-gpu"

pe = pefile.PE(os.path.join(TORCH_LIB, "torch_python.dll"), fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
deps = [e.dll.decode() for e in pe.DIRECTORY_ENTRY_IMPORT]
print("torch_python.dll imports:")
for d in deps:
    print("  ", d)

os.add_dll_directory(TORCH_LIB)
os.add_dll_directory(ENV_ROOT)
os.add_dll_directory(os.path.join(ENV_ROOT, "Library", "bin"))

print("\nTrying to load each dep individually:")
for d in deps:
    try:
        ctypes.CDLL(d)
        print(f"  OK    {d}")
    except OSError as e:
        print(f"  FAIL  {d}: {e}")
