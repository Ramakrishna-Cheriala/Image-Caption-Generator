import os
from pathlib import Path

# os.chdir("../")

CONFIG_FILE_PATH = Path(
    r"C:\Users\ramak\OneDrive\Desktop\P2\Image-Caption-Generator\config\config.yaml"
)
PARAMS_FILE_PATH = Path(
    r"C:\Users\ramak\OneDrive\Desktop\P2\Image-Caption-Generator\params.yaml"
)

# CONFIG_FILE_PATH = Path("config\config.yaml")
# PARAMS_FILE_PATH = Path("params.yaml")

# # print(CONFIG_FILE_PATH)

print(os.path.exists(CONFIG_FILE_PATH), "\n", os.path.exists(PARAMS_FILE_PATH))
