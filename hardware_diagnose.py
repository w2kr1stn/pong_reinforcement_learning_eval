import psutil
import torch


def check_hardware():
    print("=== HARDWARE DIAGNOSE ===")

    # 1. CPU Check
    cores_phys = psutil.cpu_count(logical=False)
    cores_log = psutil.cpu_count(logical=True)
    print(f"CPU: {cores_phys} Physische Kerne, {cores_log} Logische Threads")

    # 2. RAM Check
    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024**3)
    avail_ram_gb = ram.available / (1024**3)
    print(f"RAM: {total_ram_gb:.2f} GB Gesamt, {avail_ram_gb:.2f} GB Verfügbar")

    # 3. GPU / Accelerator Check
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        print("Compute Device: NVIDIA CUDA (GPU) ✅")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # VRAM Check (approx.)
        print(f"VRAM Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        print("Compute Device: Apple Metal Performance Shaders (MPS / M1/M2/M3) ✅")
        print("Hinweis: Ideal für MacBooks.")
    else:
        print("Compute Device: CPU ⚠️")
        print("Achtung: Training wird OHNE GPU sehr langsam sein (Faktor 10-50x langsamer).")


if __name__ == "__main__":
    check_hardware()
