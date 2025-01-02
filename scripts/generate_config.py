from itertools import product
import json
from dataset import CommonVoiceConfig, VoxPopuliConfig

DATASET_CONFIGS = [CommonVoiceConfig, VoxPopuliConfig]


for dataset_config in DATASET_CONFIGS:

    dataset_supported_langs = set(dataset_config.langs)

    for model, model_supported_langs in MODEL2LANG_SUPPORT.items():

        supported_langs = dataset_supported_langs.intersection(model_supported_langs)
        print(f"Supported langs for {model} on {dataset_config}: {supported_langs}")

        info = dict()

        model_sanitized = model.replace("/", "--")

        for i, (lang, split) in enumerate(
            product(supported_langs, dataset_config.splits)
        ):

            info[i] = {
                "lang": lang,
                "split": split,
                "model": model,
            }

        output_file = (
            f"config/{dataset_config.sanitized_id()}_{model_sanitized}_default.json"
        )

        with open(output_file, "w") as fp:
            json.dump(info, fp, indent=2)
