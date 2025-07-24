"""
Script that implements our sampling strategy (see README).
After N stratified sampling, it computes aggregate ASR metrics and unidirectional ttest statistics.
"""

import json
import os
import time

from tqdm import tqdm
import fire
import jiwer
import numpy as np
import pandas as pd
import scipy
from transformers import set_seed
import cyrtranslit
from typing import Union

# from mozilla_cv import LANGS_TO_LOAD_REMOTE
# from fleurs import ID_2_GENDER_MAP
from normalizers import BasicTextNormalizer, EnglishTextNormalizer
from joblib import Parallel, delayed
import logging

from config import CommonVoiceConfig, VoxPopuliConfig, MODEL2LANG_SUPPORT
from utils import log_arguments


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("transcription.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_metrics_row(metric, reference, transcription, lang, normalize: bool):
    if normalize:

        if lang == "en":
            normalizer = EnglishTextNormalizer()
        else:
            normalizer = BasicTextNormalizer()

        reference = normalizer(reference).strip()
        transcription = normalizer(transcription).strip()

    if metric == "wer":
        return jiwer.wer(reference, transcription)
    elif metric == "cer":
        return jiwer.cer(reference, transcription)
    else:
        raise ValueError(f"Unknown metric {metric}")


def compute_metrics(df, lang, whisper_normalize_text: bool = False):
    if len(df["reference"]):
        references = df["reference"].tolist()
        transcriptions = df["transcription"].tolist()

        # we strip away any parenthesis and square brackets
        # references = [re.sub(r"[\[\]()]", "", r) for r in references]
        # transcriptions = [re.sub(r"[\[\]()]", "", r) for r in transcriptions]

        if whisper_normalize_text:
            if lang == "en":
                normalizer = EnglishTextNormalizer()
            else:
                normalizer = BasicTextNormalizer()

            references = [normalizer(r).strip() for r in references]
            transcriptions = [normalizer(t).strip() for t in transcriptions]

            # if the reference is empty after the normalization (it might happen if the whole text is in between parentheses), we set it to the transcription so that wer/cer are 0.0
            references = [
                t if r == "" else r for r, t in zip(references, transcriptions)
            ]

        return {
            "wer": jiwer.wer(references, transcriptions),
            "cer": jiwer.cer(references, transcriptions),
        }
    else:
        return {"wer": np.nan, "cer": np.nan}


def find_users_below_percentile(df, minority_accept_percentile):
    spu_array = df["client_id"].value_counts()
    perc_abs_threshold = np.percentile(spu_array.values, minority_accept_percentile)
    selected_users = spu_array.loc[spu_array <= perc_abs_threshold]
    return set(selected_users.index)


def add_prefix_to_keys(d, prefix):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def add_frequency_weight(df):
    """Compute the relative frequency for each user, it'll be needed for stratification."""
    w = df["client_id"].value_counts(normalize=True)
    w.name = "weight"
    return df.join(w, on="client_id")


def sample_and_compute_metrics(
    minority_df, majority_df, minority_perc_sampled, lang, whisper_normalize_text
):
    smallest_df = minority_df if len(minority_df) < len(majority_df) else majority_df
    per_group_sample_size = int(minority_perc_sampled * len(smallest_df))
    curr_min_df = minority_df.sample(n=per_group_sample_size, weights="weight")
    curr_maj_df = majority_df.sample(n=per_group_sample_size, weights="weight")
    sample_size = 2 * per_group_sample_size

    # 2. select samples from the minority group
    # curr_min_df = minority_df.sample(frac=minority_perc_sampled, weights="weight")
    # sample_size = len(curr_min_df)

    # if len(curr_min_df) > len(majority_df):
    #     # print(
    #     #     "Sampled minority group is larger than the majority group. Downsampling..."
    #     # )
    #     curr_min_df = curr_min_df.sample(n=len(majority_df), weights="weight")
    #     sample_size = len(curr_min_df)

    # # 3. select the same number of samples from the maj grp
    # curr_maj_df = majority_df.sample(n=len(curr_min_df), weights="weight")

    maj_metrics = compute_metrics(curr_maj_df, lang, whisper_normalize_text)
    min_metrics = compute_metrics(curr_min_df, lang, whisper_normalize_text)
    mean_metrics = compute_metrics(
        pd.concat([curr_maj_df, curr_min_df]), lang, whisper_normalize_text
    )

    return {
        "min_wer": min_metrics["wer"],
        "maj_wer": maj_metrics["wer"],
        # "diff_wer": min_metrics["wer"] - maj_metrics["wer"],
        "min_cer": min_metrics["cer"],
        "maj_cer": maj_metrics["cer"],
        # "diff_cer": min_metrics["cer"] - maj_metrics["cer"],
        "subsample_wer": mean_metrics["wer"],
        "subsample_cer": mean_metrics["cer"],
        "subsample_size": sample_size,
    }


def create_output_folders(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/empty_stats", exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)


def evaluate(
    transcription_dir: str,
    lang: str,
    model: str,
    dataset_config: Union[CommonVoiceConfig, VoxPopuliConfig],
    output_dir: str,
    group_contrast: str,
    do_sampling: bool,
    apply_sampling_minority,
    apply_sampling_majority,
    split,
    num_proc,
    minority_accept_percentile,
    n_iterations,
    minority_perc_sampled,
    overwrite_results,
    # fleurs_speaker_info_dir,
    # skip_support_filter,
):
    dataset_id = dataset_config.sanitized_id()
    model_id = model.replace("/", "--")
    create_output_folders(os.path.join(output_dir, dataset_id))

    group_contrast_info = dataset_config.group_contrasts[group_contrast]
    target_col = group_contrast_info["target_col"]
    minority_group = group_contrast_info["minority_group"]
    majority_group = group_contrast_info["majority_group"]

    output_file = f"results_{model_id}_{dataset_id}_{split}_{lang}_{target_col}_{majority_group}_{minority_group}.json"
    if (
        os.path.exists(os.path.join(output_dir, dataset_id, output_file))
        and not overwrite_results
    ):
        logger.info(f"Output file {output_file} exists alread. Skipping...")
        return

    dfs = list()
    for s in dataset_config.splits:
        file_path = os.path.join(
            transcription_dir, dataset_id, lang, f"{s}_{model_id}.jsonl"
        )
        df = pd.read_json(file_path, lines=True)
        logger.info(f"Loaded from split {s}, {len(df)} transcriptions")
        df["split"] = [s] * len(df)
        dfs.append(df)
    transcription_df = pd.concat(dfs)

    # if target_col == "gender" and "fleurs" in dataset:
    #     print(f"Processing FLEURS: mapping gender IDs with {ID_2_GENDER_MAP}")
    #     transcription_df[target_col] = transcription_df[target_col].apply(
    #         lambda x: ID_2_GENDER_MAP[x]
    #     )

    # If it is a FAMA model, remove <lang:en> or <lang:it> tags
    if "FBK-MT" in model:
        transcription_df["transcription"] = transcription_df[
            "transcription"
        ].str.replace(r"<lang:[a-z]{2}>", "", regex=True)
        transcription_df["transcription"] = transcription_df["transcription"].apply(
            lambda x: x.strip()
        )

    logger.info(f"Number of transcriptions found: {len(transcription_df)}")
    logger.debug("Some transcriptions loaded")
    logger.debug(transcription_df["transcription"].iloc[:5].tolist())

    ##########
    ## Handle FLEURS specific stuff
    ##########

    # if "fleurs" in dataset:
    #     print("Processing FLEURS: Loading the speaker id info we previously computed")

    #     fleurs_speaker_info = pd.read_csv(
    #         f"{fleurs_speaker_info_dir}/speaker_info_{lang}.csv"
    #     )
    #     transcription_df["client_id"] = fleurs_speaker_info["client_id"].values
    #     # for idx, (gender_from_tr, gender_from_si) in enumerate(
    #     #     zip(transcription_df["gender"], fleurs_speaker_info["gender"])
    #     # ):
    #     #     if gender_from_tr != ID_2_GENDER_MAP[gender_from_si]:
    #     #         print(idx, gender_from_tr, ID_2_GENDER_MAP[gender_from_si])

    #     for idx, row in transcription_df.iterrows():
    #         if (row["client_id"].startswith("female") and row["gender"] == "male") or (
    #             row["client_id"].startswith("male") and row["gender"] == "female"
    #         ):
    #             raise RuntimeError(
    #                 f"(Precomputed) speaker ID and gender at {idx} do not match"
    #             )

    # if split != "all":
    #     print(f"Split is {split}. Filtering out the rest.")
    #     if split != "devtest":
    #         transcription_df = transcription_df.loc[transcription_df["split"] == split]
    #     else:
    #         transcription_df = transcription_df.loc[
    #             transcription_df["split"].isin(["validation", "test"])
    #         ]

    ############
    # VAD filter
    ############

    # if not skip_support_filter:
    #     # 3. Filter out empty recordings (VAD pipeline) We saw some issues with FLEURS es
    #     support_file = f"{os.path.dirname(output_dir)}/dataset_statistics/support_{dataset_id}_all_{lang}.csv"
    #     support_file = f"./results-interim-asr-performance-gap/dataset_statistics/support_{dataset_id}_all_{lang}.csv"
    #     print(f"Loading support file from {support_file}")
    #     support_df = pd.read_csv(support_file, index_col="rid")

    #     if split != "all":
    #         if split != "devtest":
    #             support_df = support_df.loc[support_df["split"] == split]
    #         else:
    #             support_df = support_df.loc[
    #                 support_df["split"].isin(["validation", "test"])
    #             ]

    #     mask_records_with_audio = (support_df["support"] > 0).values
    #     print(
    #         "Number of transcriptions before filtering empty records:",
    #         len(transcription_df),
    #     )
    #     print("SHAPES", transcription_df.shape, support_df.shape)
    #     transcription_df = transcription_df.loc[mask_records_with_audio]
    #     print(
    #         "Number of transcriptions after filtering empty records:",
    #         len(transcription_df),
    #     )

    # Some references might be empty due to issues in the original dataset. We filter them out.
    init_len = len(transcription_df)
    is_full = lambda x: x != None and x != ""
    transcription_df = transcription_df.loc[
        transcription_df["reference"].apply(is_full)
    ]
    final_len = len(transcription_df)
    logger.info(f"Filtering out {init_len - final_len} samples with empty references")

    # Let's also count how many empty transcriptions we have. But we do not filter them out.
    empty_transcriptions = len(
        transcription_df.loc[~transcription_df["transcription"].apply(is_full)]
    )
    logger.debug(
        f"Empty transcriptions found: {empty_transcriptions}",
    )
    transcription_df["transcription"] = transcription_df["transcription"].fillna("")

    empty_stats = {
        "reference": final_len - init_len,
        "transcription": empty_transcriptions,
    }
    with open(
        os.path.join(
            output_dir,
            dataset_id,
            "empty_stats",
            f"empty_stats_{model_id}_{dataset_id}_{split}_{lang}_{target_col}_{majority_group}_{minority_group}.json",
        ),
        "w",
    ) as fp:
        json.dump(empty_stats, fp, indent=2)

    # If it's serbian or russian, transliterate everything into cyrillic
    if lang == "sr" or lang == "ru":
        logger.info(f"Transliterating to cyrillic {lang}")
        transcription_df["transcription"] = transcription_df["transcription"].apply(
            lambda x: cyrtranslit.to_cyrillic(x, lang)
        )
        transcription_df["reference"] = transcription_df["reference"].apply(
            lambda x: cyrtranslit.to_cyrillic(x, lang)
        )

    logger.info("GENDER DISTRIBUTION IN THE DATA CONSIDERED")
    logger.info(transcription_df[target_col].value_counts())

    # Separate majority and minority groups
    minority_df = transcription_df.loc[transcription_df[target_col] == minority_group]
    logger.debug(
        f"Unique minority speakers count: {minority_df.client_id.unique().size}"
    )
    logger.debug(f"Some speaker ids: {minority_df.client_id.unique()[:5]}")
    majority_df = transcription_df.loc[transcription_df[target_col] == majority_group]
    logger.debug(
        f"Unique majority speakers count: {majority_df.client_id.unique().size}"
    )
    logger.debug(f"Some speaker ids: {majority_df.client_id.unique()[:5]}")

    # 6. compute metrics on the whole split
    whisper_normalize_text = "whisper" in model
    logger.info(
        f"Using the normalization strategy from Whisper's code: {whisper_normalize_text}"
    )
    results = dict()

    results["model_id"] = model_id
    results["dataset_id"] = dataset_id
    results["lang"] = lang
    results["split"] = split
    results["minority_group"] = minority_group
    results["majority_group"] = majority_group
    results["target_col"] = target_col
    results["eval_metric"] = "cer" if lang in ["yo", "ja"] else "wer"

    full_metrics = compute_metrics(transcription_df, lang, whisper_normalize_text)
    results |= add_prefix_to_keys(full_metrics, "presample")

    results |= add_prefix_to_keys(
        compute_metrics(majority_df, lang, whisper_normalize_text), "presample_maj"
    )
    results |= add_prefix_to_keys(
        compute_metrics(minority_df, lang, whisper_normalize_text), "presample_min"
    )
    results["presample_diff_wer"] = (
        results["presample_min_wer"] - results["presample_maj_wer"]
    )
    results["presample_diff_cer"] = (
        results["presample_min_cer"] - results["presample_maj_cer"]
    )

    logger.info(f"do_sampling set to {do_sampling}!")

    if do_sampling:
        logger.info(f"Enabling bootstrap sampling for {n_iterations} iterations")

        if apply_sampling_minority:
            logger.info("Applying sampling to the minority group")
            mino_users = find_users_below_percentile(
                minority_df, minority_accept_percentile
            )
        else:
            mino_users = minority_df["client_id"].unique()
        if apply_sampling_majority:
            logger.info("Applying sampling to the majority group")
            majo_users = find_users_below_percentile(
                majority_df, minority_accept_percentile
            )
        else:
            majo_users = majority_df["client_id"].unique()

        minority_df = minority_df.loc[minority_df["client_id"].isin(mino_users)]
        majority_df = majority_df.loc[majority_df["client_id"].isin(majo_users)]
        logger.debug(f"Selected {len(mino_users)} users from the minority group")
        logger.debug(f"Selected {len(majo_users)} users from the majority group")
        logger.debug(f"{len(minority_df)} records from minority users")
        logger.debug(f"{len(majority_df)} records from majority users")

        minority_df, majority_df = map(add_frequency_weight, (minority_df, majority_df))
        min_count, maj_count = len(minority_df), len(majority_df)

        results["largest_group"] = "minority" if min_count > maj_count else "majority"

        # We do not subsample any group (see details in the paper)
        # overall_maj = majority_df.sample(n=min(min_count, maj_count), weights="weight")
        # overall_min = minority_df.sample(n=min(min_count, maj_count), weights="weight")
        overall_maj = majority_df
        overall_min = minority_df
        logger.debug(
            f"Number of records after subsampling (majority): {len(overall_maj)}"
        )
        logger.debug(
            f"Number of records after subsampling (minority): {len(overall_min)}"
        )

        # Compute sentence-level metrics
        sample_df = pd.concat([overall_maj, overall_min])
        wers = list()
        cers = list()
        for _, row in sample_df.iterrows():
            wers.append(
                compute_metrics_row(
                    "wer",
                    row["reference"],
                    row["transcription"],
                    lang,
                    whisper_normalize_text,
                )
            )
            cers.append(
                compute_metrics_row(
                    "cer",
                    row["reference"],
                    row["transcription"],
                    lang,
                    whisper_normalize_text,
                )
            )
        sample_df["wer"] = wers
        sample_df["cer"] = cers

        sample_df.to_csv(
            os.path.join(
                output_dir,
                dataset_id,
                "samples",
                f"sample_{model_id}_{dataset_id}_{split}_{lang}_{target_col}_{majority_group}_{minority_group}.csv",
            )
        )
        results |= add_prefix_to_keys(
            compute_metrics(
                pd.concat([overall_maj, overall_min]), lang, whisper_normalize_text
            ),
            "overall",
        )

        # 2. compute metrics on all minority vs majority (we subsample to the less numerous among the two)
        results |= add_prefix_to_keys(
            compute_metrics(overall_maj, lang, whisper_normalize_text), "overall_maj"
        )
        results |= add_prefix_to_keys(
            compute_metrics(overall_min, lang, whisper_normalize_text), "overall_min"
        )
        results["overall_diff_wer"] = (
            results["overall_min_wer"] - results["overall_maj_wer"]
        )
        results["overall_diff_cer"] = (
            results["overall_min_cer"] - results["overall_maj_cer"]
        )
        results["overall_sample_size"] = len(overall_maj)

        samples = Parallel(n_jobs=num_proc, verbose=10)(
            delayed(sample_and_compute_metrics)(
                overall_min,
                overall_maj,
                minority_perc_sampled,
                lang,
                whisper_normalize_text,
            )
            for _ in range(n_iterations)
        )

        stats = pd.DataFrame(samples)  # type: ignore
        mean_stats = stats.mean()

        # Average results across all samples, separately by group
        results["mean_maj_wer"] = mean_stats["maj_wer"]
        results["mean_min_wer"] = mean_stats["min_wer"]
        results["mean_maj_cer"] = mean_stats["maj_cer"]
        results["mean_min_cer"] = mean_stats["min_cer"]

        def aggregate_sample_statistics(metric: str):
            diff_ = stats[f"min_{metric}"] - stats[f"maj_{metric}"]
            diff_rel_ = stats[f"min_{metric}"] / stats[f"maj_{metric}"] - 1
            return {
                f"{metric}_diff_mean": diff_.mean(),
                f"{metric}_diff_std": diff_.std(),
                f"{metric}_diff_rel_mean": diff_rel_.mean(),
                f"{metric}_diff_rel_std": diff_rel_.std(),
            }

        results |= aggregate_sample_statistics("wer")
        results |= aggregate_sample_statistics("cer")

        # results["mean_diff_wer"] = mean_stats["min_wer"] - mean_stats["maj_wer"]
        # results["mean_diff_cer"] = mean_stats["min_cer"] - mean_stats["maj_cer"]

        # Statistics on the entire subsample
        results["subsample_size"] = samples[0]["subsample_size"]
        results["mean_subsample_wer"] = mean_stats["subsample_wer"]
        results["mean_subsample_cer"] = mean_stats["subsample_cer"]

        # one-side t-tests on WER and CER
        ttest_res = scipy.stats.ttest_ind(
            stats["min_wer"], stats["maj_wer"], alternative="greater"
        )
        results["1sided_ttest_wer_p"] = ttest_res.pvalue
        results["1sided_ttest_wer_stat"] = ttest_res.statistic
        ttest_res = scipy.stats.ttest_ind(
            stats["min_cer"], stats["maj_cer"], alternative="greater"
        )
        results["1sided_ttest_cer_p"] = ttest_res.pvalue
        results["1sided_ttest_cer_stat"] = ttest_res.statistic

        ttest_two_sided = scipy.stats.ttest_ind(stats["min_wer"], stats["maj_wer"])
        results["2sided_ttest_wer_p"] = ttest_two_sided.pvalue
        results["2sided_ttest_wer_stat"] = ttest_two_sided.statistic
        ttest_two_sided = scipy.stats.ttest_ind(stats["min_cer"], stats["maj_cer"])
        results["2sided_ttest_cer_p"] = ttest_two_sided.pvalue
        results["2sided_ttest_cer_stat"] = ttest_two_sided.statistic

        # info stats
        results["minority_accept_percentile"] = minority_accept_percentile
        results["n_iterations"] = n_iterations
        results["minority_perc_sampled"] = minority_perc_sampled

    with open(os.path.join(output_dir, dataset_id, output_file), "w") as fp:
        json.dump(results, fp, indent=2)

    # print("#### BASIC STATISTICS")

    # print("Mean")
    # print(stats.mean())
    # print("STD")
    # print(stats.std())

    # print("Student's t test on WER")
    # ttest_res = scipy.stats.ttest_ind(stats["min_wer"], stats["maj_wer"])
    # print("Statistics", ttest_res.statistic, "p", ttest_res.pvalue)
    # print("Confidence level (95%) of the difference in population means")
    # ci = ttest_res.confidence_interval()
    # print("Low", ci.low, "High", ci.high)


@log_arguments
def main(
    transcription_dir: str,
    model: str,
    dataset: str,
    output_dir: str,
    group_contrast: str,
    do_sampling: bool = True,
    apply_sampling_minority: bool = False,
    apply_sampling_majority: bool = False,
    split: str = "devtest",
    num_proc: int = 2,
    minority_accept_percentile: float = 99,
    n_iterations: int = 1500,
    minority_perc_sampled: float = 0.4,
    # load_type: str = "local",
    # reference_col: str = "sentence",
    overwrite_results: bool = False,
    fleurs_speaker_info_dir: str = None,
    skip_support_filter: bool = False,
):
    set_seed(42)

    lang_supported_model = set(MODEL2LANG_SUPPORT[model])
    dataset_config = CommonVoiceConfig if "cv" in dataset else VoxPopuliConfig
    lang_supported_dataset = set(dataset_config.langs)

    lang_to_eval = lang_supported_model.intersection(lang_supported_dataset)
    logger.info(f"Languages to evaluate: {lang_to_eval}")

    for lang in tqdm(lang_to_eval, desc="Language"):
        evaluate(
            transcription_dir=transcription_dir,
            lang=lang,
            model=model,
            dataset_config=dataset_config,
            output_dir=output_dir,
            group_contrast=group_contrast,
            do_sampling=do_sampling,
            apply_sampling_minority=apply_sampling_minority,
            apply_sampling_majority=apply_sampling_majority,
            split=split,
            num_proc=num_proc,
            minority_accept_percentile=minority_accept_percentile,
            n_iterations=n_iterations,
            minority_perc_sampled=minority_perc_sampled,
            overwrite_results=overwrite_results,
            # fleurs_speaker_info_dir=fleurs_speaker_info_dir,
            # skip_support_filter=skip_support_filter,
        )


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed {int(time.time() - stime)} seconds")
