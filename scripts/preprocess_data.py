from __future__ import annotations 

import argparse
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm.rich import trange


from ai_alignment_linear_model_extension.preprocessing.soc_format_preprocessing import SOCFilePreprocessor
# from rich.default_styles import args

def main():
    """
    Main function to preprocess a .soc file and generate a JSON dictionary.
    """

    # parser = argparse.ArgumentParser(description="Preprocess a .soc file to extract the number of alternatives and voters.")
    # parser.add_argument("soc_file_path", type=str, help="Path to the .soc file.")
    # parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Sentence transformer model to use for encoding alternative descriptions.")
    # args = parser.parse_args()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    data_path = Path("data/raw/00070_habermas/")
    output_path = Path("data/processed/00070_habermas/")
    # data_file_name = "00070-00000001.json"
    if Path(output_path).exists() is False:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    soc_file_paths = list(data_path.glob("*.soc"))

    for i in trange(len(soc_file_paths), desc="Preprocessing .soc files"):
        soc_file_path = soc_file_paths[i]
        data_file_name = f"{soc_file_path.stem}.json"
        json_path = Path(f"{output_path}/{data_file_name}")
        if json_path.exists():
            print(f"Preprocessed data already exists for {soc_file_path}. Skipping.")
            continue

        preprocessor = SOCFilePreprocessor(soc_file_path, model=model)
        json_dict = preprocessor.generate_json_dict()

        print(f"Preprocessed data for {soc_file_path.name} to be saved to {json_path.name}")

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=4)
        print(f"Preprocessed data saved to {json_path}")


if __name__ == "__main__":
    main()