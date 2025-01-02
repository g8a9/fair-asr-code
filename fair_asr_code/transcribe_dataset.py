import gc
import csv
import json
import logging
import math
import os
import time

import fire
import librosa
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from datasets import Dataset, load_dataset
from tqdm import tqdm
from utils import log_arguments

# from fleurs import LANG_TO_CONFIG_MAPPING
from datasets import Audio

# from codecarbon import track_emissions
from codecarbon import EmissionsTracker

from transcriber import HfTranscriber, NeMoTranscriber

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("transcription.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# @track_emissions
@log_arguments
def transcribe(
    dataset_id: str,
    lang: str,
    split: str,
    model_name_or_path: str,
    output_dir: str,
    input_dir: str = None,
    # model: str,
    dry_run: bool = False,
    batch_size: int = 1,
    num_workers: int = 1,
    overwrite_output: bool = False,
    chunk_size: int = 3000,
    # column names to change depending on the dataset used
    reference_col: str = "sentence",
    speaker_id_col: str = "client_id",
    gender_col: str = "gender",
    target_sr: int = 16_000,
    max_length_seconds: int = 30,
    load_and_resample: bool = False,
):

    if dataset_id == "cv_17":
        logger.info("Starting to load and decode local audio files.")
        split_file = f"{input_dir}/transcript/{lang}/{split}.tsv"
        df = pd.read_csv(split_file, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8")
        df["audio_path"] = df["path"].apply(
            lambda x: f"{input_dir}/audio/{lang}/{split}/{lang}_{split}_0/{x}"
        )
        data = Dataset.from_pandas(df)
        logger.info(f"Created a HF dataset with {len(data)} samples.")

        # Measure the sampling rate of the first audio in data
        first_audio_path = data["audio_path"][0]
        _, original_sr = librosa.load(first_audio_path, sr=None)
        logger.info(f"Sampling rate of the first audio: {original_sr}")

        if original_sr != target_sr:
            logger.warning(
                f"Sampling rate of the audio files is not {target_sr}. Resampling is required."
            )
            load_and_resample = True

        # Measure the sampling rate of the first audio in data
        first_audio_path = data["audio_path"][0]
        _, original_sr = librosa.load(first_audio_path, sr=None)
        logger.info(f"Sampling rate of the first audio: {original_sr}")
    elif dataset_id == "facebook/voxpopuli":
        data = load_dataset(
            dataset_id,
            lang,
            split=split,
            num_proc=num_workers,
            trust_remote_code=True,
        )
        data = data.cast_column("audio", Audio(sampling_rate=16000))
    else:
        raise ValueError("Unknown dataset")

    data = data.add_column("rid", [f"{split}_{i}" for i in range(len(data))])  # type: ignore
    logger.info(
        f"Added a unique id column to the dataset, initial_value: {data['rid'][0]}, final value: {data['rid'][-1]}"
    )

    # clean_model_name = model.replace("/", "--")
    # clean_dataset_name = dataset.replace("/", "--")
    # load_type = config["load_type"]
    # local_dir = config.get("local_dir", None)
    # reference_col = config.get("reference_col", reference_col)
    # speaker_id_col = config.get("speaker_id_col", speaker_id_col)

    # process_slice = config.get("process_slice", None)
    # if process_slice:
    #     print(f"### Slicing the dataset wiht slice: {process_slice}")

    # out_file = (
    #     (f"{output_dir}/{clean_model_name}_{clean_dataset_name}_{split}_{lang}.tsv")
    #     if not process_slice
    #     else (
    #         f"{output_dir}/{clean_model_name}_{clean_dataset_name}_{split}_{lang}_{process_slice}.tsv"
    #     )
    # )
    # if os.path.exists(out_file) and not overwrite_output:
    #     print(f"Output file {out_file} exists already. Skipping...")
    #     return

    # lang_code = lang if "fleurs" not in dataset else LANG_TO_CONFIG_MAPPING[lang]

    # if load_type == "remote":
    #     print("Loading remote dataset.", dataset, lang_code, split, num_workers)
    #     data = load_dataset(
    #         dataset,
    #         lang_code,
    #         split=split,
    #         num_proc=num_workers,
    #         trust_remote_code=True,
    #     )

    #     if "mozilla" in dataset:
    #         data = data.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))

    # elif load_type == "local":

    logger.info("Loading finished.")

    # else:
    # raise ValueError("Load type unknown")

    logger.info(f"Initial len {len(data)}")

    if dry_run:
        print("Running dry run (first 128 samples)...")
        data = data.select(range(128))

    # if process_slice:
    #     left_p, right_p = process_slice.split("-")
    #     left_idx = int(float(left_p) * len(data))
    #     right_idx = int(float(right_p) * len(data))
    #     print(
    #         f"Processing only data between {left_p} (idx: {left_idx}) and {right_p} (idx: {right_idx})"
    #     )
    #     data = data.select(range(left_idx, right_idx))  # type: ignore

    #####
    # Prepare the pipeline and model
    #####
    if model_name_or_path == "nvidia/canary-1b":
        transcriber = NeMoTranscriber("nvidia/canary-1b", lang)
    else:
        transcriber = HfTranscriber(
            model_name_or_path=model_name_or_path,
            tgt_lang=lang,
            torch_dtype=torch.bfloat16,
            # chunk_length_s=30,
            device="cuda",
        )
    logger.info("Transcriber loaded")

    def load_and_resample_to_khz(examples):
        resampled = list()
        is_valid = list()
        lengths = list()
        for path in examples["audio_path"]:
            try:
                arr, _ = librosa.load(path, sr=target_sr)
                resampled.append(arr)
                is_valid.append(True)
                lengths.append(arr.shape[0])
            except Exception as e:
                logger.warning(
                    f"Error while decoding and/or resampling. Skipping file: {path}",
                    exc_info=1,
                )
                # print(e)
                resampled.append(np.zeros((1,), dtype=np.float32))
                is_valid.append(False)
                lengths.append(-1)
        return {"audio": resampled, "is_valid": is_valid, "length": lengths}

    def transcribe_chunk(data, transcriber, load_and_resample: bool):
        """
        CV data is in 48 kHz, VP and FLEURS are in 16 kHz.
        Whisper and Seamless require 16 kHZ sampled data.
        """

        if load_and_resample:
            # 1. decode and resample to the target sampling rate
            stime = time.time()
            data = data.map(
                load_and_resample_to_khz,
                batched=True,
                num_proc=num_workers,
                desc="Loading and resampling to 16khz",
            )
            # 2. filter out the invalid records
            data = data.filter(
                lambda examples: examples["is_valid"],
                batched=True,
                num_proc=num_workers,
                desc="Filter valid records",
            )

            logger.debug(f"Time to load, resample, and filter:", time.time() - stime)

            raw_audio = data["audio"]
            chunk_max_length = max(data["length"])
        else:
            raw_audio = [d["array"] for d in data["audio"]]
            chunk_max_length = max([a.shape[0] for a in raw_audio])

        # 2. get the transcriptions
        logger.info(f"Max length (s) found in chunk: {chunk_max_length / target_sr}")
        logger.info(f"Trimming to: {max_length_seconds}")
        if chunk_max_length / target_sr > max_length_seconds:
            logger.warning("SOME AUDIO WILL GET TRIMMED")

        logger.debug(f"Transcribing a chuck of {len(raw_audio)} samples.")
        logger.debug("Some references", data[reference_col][:3])

        targs = {
            "raw_audio": raw_audio,
            "batch_size": batch_size,
            "show_progress_bar": True,
        }
        if not isinstance(transcriber, NeMoTranscriber):
            targs |= {
                "num_workers": num_workers,
                "max_length": max_length_seconds * target_sr,
                "sampling_rate": target_sr,
            }

        transcriptions = transcriber(**targs)

        try:
            # FLEURS has no speaker id
            speaker_ids = data[speaker_id_col]
        except:
            speaker_ids = [None] * len(data)

        return {
            "reference": data[reference_col],
            "transcription": transcriptions,
            "client_id": speaker_ids,
            "gender": data[gender_col],
            "rid": data["rid"],
        }

    # split decoding + transcribing in chunks
    # if enable_chunk_decoding:
    idxs = np.arange(len(data))
    n_chunks = math.ceil(len(data) / chunk_size)
    batches = np.array_split(idxs, n_chunks)

    transcriptions = list()
    references = list()
    clients = list()
    genders = list()
    rids = list()
    for batch in tqdm(batches, desc="Chunk", total=n_chunks):
        curr_data = data.select(batch)
        # curr_data = curr_data.cast_column("audio", Audio()) # loading demanded to HF
        r = transcribe_chunk(
            curr_data,
            transcriber,
            load_and_resample=load_and_resample,  # (load_type == "local" and "mozilla" in dataset),
        )

        logger.info("Verify")
        logger.info(r["transcription"][:3])
        logger.info(r["reference"][:3])

        transcriptions.extend(r["transcription"])
        references.extend(r["reference"])
        clients.extend(r["client_id"])
        genders.extend(r["gender"])
        rids.extend(r["rid"])
        gc.collect()

    # else:
    # r = transcribe_chunk(data, transcriber)
    # transcriptions = r["transcription"]
    # references = r["reference"]
    # clients = r["client_id"]
    # genders = r["gender"]
    # rids = r["rid"]

    result = {
        "rid": rids,
        "client_id": clients,
        "gender": genders,
        "transcription": transcriptions,
        "reference": references,
    }

    model_name_sanizited = model_name_or_path.replace("/", "--")
    out_file = os.path.join(
        output_dir, dataset_id, lang, f"{split}_{model_name_sanizited}.jsonl"
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    results = pd.DataFrame(result).set_index("rid")
    results.to_json(out_file, orient="records", lines=True, force_ascii=False)


def main(
    dataset_id: str,
    config_file: str,
    config_id: int,
    output_dir: str,
    input_dir: str = None,
    dry_run: bool = False,
    batch_size: int = 1,
    num_workers: int = 1,
    overwrite_output: bool = False,
    chunk_size: int = 3000,
    reference_col: str = "sentence",
    speaker_id_col: str = "client_id",
    load_and_resample: bool = False,
):
    with open(config_file) as fp:
        configs = json.load(fp)

    config = configs[str(config_id)]
    logger.info("Loaded config:")
    logger.info(config)

    lang = config["lang"]
    split = config["split"]
    model = config["model"]
    dataset = dataset_id

    model_sanitized = model.replace("/", "--")
    os.makedirs(os.path.join(output_dir, "emissions"), exist_ok=True)

    # with EmissionsTracker(
    #     log_level="debug",
    #     output_dir=os.path.join(output_dir, "emissions"),
    #     output_file=f"{dataset}_{lang}_{split}_{model_sanitized}.csv",
    # ) as tracker:
    transcribe(
        dataset_id=dataset,
        lang=lang,
        split=split,
        model_name_or_path=model,
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=dry_run,
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite_output=overwrite_output,
        chunk_size=chunk_size,
        reference_col=reference_col,
        speaker_id_col=speaker_id_col,
        load_and_resample=load_and_resample,
    )


if __name__ == "__main__":
    stime = time.time()
    logger.info("RUN STARTED")
    fire.Fire(main)
    logger.info(f"RUN COMPLETED ({time.time() - stime:.0f} SECONDS)")
