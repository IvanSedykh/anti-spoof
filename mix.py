from pathlib import Path

from fire import Fire

from src.mixing.seminar_code import MixtureGenerator, LibriSpeechSpeakerFiles


def main(
        data_dir: str,
        out_folder: str,
        n_files: int,
        num_speakers: int = None,
        snr_levels: list[int] = None,
        test: bool = False,
        num_workers: int = 1
        ):
    if snr_levels is None:
        snr_levels = [0]

    # print args
    print("data_dir:", data_dir)
    print("out_folder:", out_folder)
    print("n_files:", n_files)
    print("num_speakers:", num_speakers)
    print("snr_levels:", snr_levels)
    print("test:", test)
    print("num_workers:", num_workers)

    
    data_dir = Path(data_dir)


    speaker_files = []
    for speaker_path in data_dir.iterdir():
        speaker_id = speaker_path.name
        files = LibriSpeechSpeakerFiles(
            speaker_id=speaker_id,
            audios_dir=data_dir,
            audioTemplate="*.flac"
            )
        speaker_files.append(files)

    if num_speakers is not None:
        # sort by number of files
        speaker_files = sorted(speaker_files, key=lambda x: len(x.files), reverse=True)
        speaker_files = speaker_files[:num_speakers]


    mg = MixtureGenerator(
        speakers_files=speaker_files,
        out_folder=out_folder,
        nfiles=n_files,
        test=test
    )

    mg.generate_mixes(
        snr_levels=snr_levels,
        num_workers=num_workers,
        trim_db=20,
        vad_db=20,
        audioLen=3
    )

if __name__ == '__main__':
    Fire(main)