"""Config file for the dataset in the framework."""

from abc import ABC, abstractmethod


class BaseDatasetConfig(ABC):

    def __repr__(self):
        return self.dataset_id

    @classmethod
    def sanitized_id(self):
        return self.dataset_id.replace("/", "--")


class CommonVoiceConfig(BaseDatasetConfig):
    name = "Common Voice 17"
    dataset_id = "cv_17"
    splits = ["dev", "test"]
    speaking_condition = "read"

    group_contrasts = {
        "F-M": {
            "target_col": "gender",
            "minority_group": "female_feminine",
            "majority_group": "male_masculine",
        }
    }

    # fmt: off
    langs = [
        "de", "en", "nl",  # Germanic
        "ru", "sr", "cs", "sk",  # Slavic
        "it", "fr", "es", "ca", "pt", "ro",  # Romance
        "sw",  # Bantu
        "yo",  # Niger-Congo
        "ja",  # Japonic
        "hu", "fi",  # Uralic
        "ar"  # Semitic
    ]
    # fmt: on


class VoxPopuliConfig(BaseDatasetConfig):
    name = "VoxPopuli"
    dataset_id = "facebook/voxpopuli"
    splits = ["validation", "test"]
    speaking_condition = "spontaneous"

    group_contrasts = {
        "F-M": {
            "target_col": "gender",
            "minority_group": "female",
            "majority_group": "male",
        }
    }

    langs = [
        "de",
        "en",
        "nl",
        "it",
        "fr",
        "es",
        "hu",
        "fi",
        "ro",
        "cs",
        "sk",
    ]


ALL_DATASET_CONFIGS = [CommonVoiceConfig, VoxPopuliConfig]

# List the largest set of languages that each model supports
MODEL2LANG_SUPPORT = {
    "openai/whisper-large-v3": CommonVoiceConfig.langs,
    "openai/whisper-large-v3-turbo": CommonVoiceConfig.langs,
    "facebook/seamless-m4t-v2-large": CommonVoiceConfig.langs,
    "nvidia/canary-1b": ["en", "de", "fr", "es"],
}
