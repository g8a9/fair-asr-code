from itertools import product
import json
from common_voice import LANGS as CV_LANGS
from common_voice import SPLITS as CV_SPLITS

DATASETS = ["cv_17"]

# List the largest set of languages that each model supports
MODEL2LANG_SUPPORT = {
    "openai/whisper-large-v3": CV_LANGS,
    "openai/whisper-large-v3-turbo": CV_LANGS,
    "facebook/seamless-m4t-v2-large": CV_LANGS,
    # "nvidia/canary-1b": ["en", "de", "fr", "es"],
}


info = dict()
for dataset in DATASETS:

    for model in MODEL2LANG_SUPPORT.keys():

        model_sanitized = model.replace("/", "--")

        # Create the config file for CV
        if dataset == "cv_17":

            for i, (lang, split) in enumerate(product(CV_LANGS, CV_SPLITS)):

                if lang not in MODEL2LANG_SUPPORT[model]:
                    continue

                info[i] = {
                    "lang": lang,
                    "split": split,
                    "model": model,
                }

        output_file = f"config/{dataset}_{model_sanitized}_default.json"

        with open(output_file, "w") as fp:
            json.dump(info, fp, indent=2)
