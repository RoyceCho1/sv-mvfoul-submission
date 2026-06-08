# Data Access

This project uses the SoccerNet-MVFoul video dataset. The dataset is not included in this repository because it is large and distributed under SoccerNet's data access terms.

## Required Dataset

Dataset:

```text
SoccerNet-MVFoul
```

Expected local path after download:

```text
data/SoccerNet/mvfouls/
```

Expected directory layout:

```text
data/SoccerNet/mvfouls/
  Train/
    annotations.json
    action_{id}/
      clip_0.mp4
      clip_1.mp4
      ...
  Valid/
    annotations.json
    action_{id}/
      clip_0.mp4
      clip_1.mp4
      ...
  Test/
    annotations.json
    action_{id}/
      clip_0.mp4
      clip_1.mp4
      ...
```

The scripts assume this path by default:

```bash
--data-root data/SoccerNet/mvfouls
```

## Official-Target Filtering

For the submitted experiments, the code follows the official target filtering used during development:

```text
action_class != "Dont know"
Offence != "Between"
Severity not in {"2.0", "4.0"}
```

Expected filtered action counts:

| Split | Count |
|---|---:|
| Train | 2,319 official-target actions / 5,277 view samples |
| Valid | 321 actions |
| Test | 247 actions |

## Download

Use the official SoccerNet download/access instructions for SoccerNet-MVFoul. If the local videos are missing, the provided helper script can be used as a starting point:

```bash
python download_mvfoul_720p.py
```

Depending on SoccerNet account permissions and token setup, the TA may need to authenticate with SoccerNet before downloading the dataset.

## Reproduction Note

Full training reproduction requires the full Train split. The final submitted reproduction target is evaluation from the provided LoRA adapter and the Valid/Test evaluation splits. If the dataset cannot be redistributed directly in the LMS zip, it must be downloaded separately and placed at the path shown above.
