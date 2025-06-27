# soccer-player-re-identification

The goal is to locate players in the clip and assign an ID to them based on the initial few frames, and assign the same ID for a player in case they leave the frame and re-appear.

For custom implentation, the combination of jersey number and jersey color should provide a unique identification for every player on the field.

# Methods used:

1. DeepSort with slight modification: ID assignment by comparing centres of previous box
2. ByteTrack implementation
3. A custom color detection + jersey number detection from bounding box of player using Optical Character Recognition (easyOCR)

# How to run:

Creating the environment:

```
python3.10 -m venv venv
```

Activating the environment:
```
./venv/scripts/activate
```

Installing requirements:
```
pip install -r requirements.txt
```

Running the Deepsort implementation:
```
python ./implementations/deepsort_version.py
```

Running the custom color detection + OCR version
```
python ./implementations/color_ocr_version.py
```

Running the ByteTrack implementation
```
python ./implementations/bytetrack_version.py
```

Once execution is completed, output is saved in outputs folder
