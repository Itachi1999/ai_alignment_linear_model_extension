from __future__ import annotations
from collections import defaultdict

from preflibtools.instances import OrdinalInstance
from pathlib import Path
import uuid
from sentence_transformers import SentenceTransformer


class SOCFilePreprocessor:
    """
    Preprocesses a .soc file to extract the number of alternatives and voters.
    """

    def __init__(self, soc_file_path: str, model: SentenceTransformer) -> None:
        self._soc_file_path = Path(soc_file_path)
        self.instance = OrdinalInstance()
        self.instance.parse_file(self._soc_file_path)
        self.model = model

    def generate_json_dict(self):
        """
        Generates a JSON dictionary containing the number of alternatives and voters.
        """
        num_alternatives = self.instance.num_alternatives
        num_voters = self.instance.num_voters

        title = self.instance.title
        if not title:
            title = f"preflib_soc_{uuid.uuid4()}"

        alternatives_dict = self.instance.alternatives_name 
        alternative_id_to_uuid = {
            real_id: str(uuid.uuid4()) 
            for real_id in alternatives_dict.keys()
        }

        alternative_description_dict = {
            alternative_id_to_uuid[real_id]: f"Question: {title}. Answer: {description}"
            for real_id, description in alternatives_dict.items()
        }

        preference_matrix = []
        for preference, count in self.instance.vote_map().items():
            # print(f"Ranking: {preference, type(preference)}, Count: {count, type(count)}")
            for _ in range(int(count)):
                ranking = []
                # print(f"Ranking: {preference, type(preference)}")
                for alt_tuple in preference:
                    # print(f"Alternative Tuple: {alt_tuple, type(alt_tuple)}")
                    alt_id = alt_tuple[0]
                    # print(f"Alternative ID: {alt_id}, UUID: {alternative_id_to_uuid[alt_id]}")
                    ranking.append(alternative_id_to_uuid[alt_id])
                preference_matrix.append(tuple(ranking))

        alternative_feature_vectors = defaultdict(list)
        for alt_uuid, description in alternative_description_dict.items():
            feature_vector = self.model.encode(description, normalize_embeddings=True)
            alternative_feature_vectors[alt_uuid] = feature_vector.tolist()


        return {
            "num_alternatives": num_alternatives,
            "num_voters": num_voters,
            "title": title,
            "alternatives": alternative_description_dict,
            "preference_matrix": tuple(preference_matrix),
            "number_of_unique_rankings": self.instance.num_unique_orders,
            "alternative_feature_vectors": dict(alternative_feature_vectors)
        }
    

        