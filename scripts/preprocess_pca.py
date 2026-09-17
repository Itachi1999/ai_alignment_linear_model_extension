from pathlib import Path
import json
import numpy as np
from tqdm.rich import trange
from sklearn.decomposition import PCA


def fit_global_pca(
    input_dir: Path,
    n_components: int,
) -> PCA:

    vectors = []
    for json_file in sorted(input_dir.glob("*.json")):
        with json_file.open("r") as file:
            data = json.load(file)

        vectors.extend(
            data["alternative_feature_vectors"].values()
        )

    X = np.asarray(vectors, dtype=np.float32)

    if X.shape[0] < n_components:
        raise ValueError(
            f"Need at least {n_components} samples, "
            f"but found {X.shape[0]}."
        )

    pca = PCA(
        n_components=n_components,
        svd_solver="full",
    )
    pca.fit(X)

    return pca


def transform_and_save(
    input_dir: Path,
    output_dir: Path,
    pca: PCA,
    dimension: int,
) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    json_file_paths = list(sorted(input_dir.glob("*.json")))
    for i in trange(len(json_file_paths), desc= f"PCA Transform for dimension {dimension}"):
        json_file = json_file_paths[i]
        with json_file.open("r") as file:
            data = json.load(file)

        feature_vectors = data["alternative_feature_vectors"]
        alternative_ids = list(feature_vectors.keys())

        X = np.asarray(
            [feature_vectors[alt_id] for alt_id in alternative_ids],
            dtype=np.float32,
        )

        X_reduced = pca.transform(X)

        data["alternative_feature_vectors"] = {
            alt_id: X_reduced[i].tolist()
            for i, alt_id in enumerate(alternative_ids)
        }

        output_file = Path(f"{output_dir}/{json_file.name}")

        with output_file.open("w") as file:
            json.dump(
                data,
                file,
                indent=4,
            )


def main():

    input_dir: Path = Path("data/processed/00070_habermas/")
    output_root: str = "data/processed/"
    dimensions: tuple[int, ...] = (150, 200, 250, 300)

    for dimension in dimensions:
        
        print(f"Fitting PCA with {dimension} components...")

        pca = fit_global_pca(
            input_dir=input_dir,
            n_components=dimension,
        )

        output_dir = Path(f"{output_root}/00070_habermas_pca_{dimension}")

        transform_and_save(
            input_dir=input_dir,
            output_dir=output_dir,
            pca=pca,
            dimension=dimension
        )

        print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()