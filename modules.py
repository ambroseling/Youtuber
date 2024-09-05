import importlib

modules = [
    "os", "numpy", "srt", "re", "ffmpeg", "whisperx", "gc", "soundfile", "scipy.io.wavfile",
    "groq", "transformers", "torch", "dotenv", "diffusers", "subprocess", "nvidia.cublas.lib", "nvidia.cudnn.lib"
]

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"Module '{module_name}' is available.")
    except ImportError:
        print(f"Module '{module_name}' is NOT available.")

for module in modules:
    # Just try to import the full module name directly
    try:
        check_module(module)
    except Exception as e:
        print(f"An error occurred while checking '{module}': {e}")
