# Smart Surroundings Assistant

AI system that answers "what can I use to do X?" by scanning your surroundings using NLP + YOLOv8 object detection.

## How it works
1. User uploads an image of surroundings
2. User types a question like "What can I use to cut something?"
3. spaCy extracts the verb → "cut"
4. Dictionary maps "cut" → ["knife", "scissors", "blade"]
5. YOLOv8 scans the image for those objects
6. System answers: "Yes! You can use the scissors to cut!"

## Team
- Person 1 — NLP module (spaCy)
- Person 2 — Vision module (YOLOv8)
- Person 3 — Dictionary, Pipeline, Answer Generator

## How to run
1. Open main.py in Google Colab
2. Run: `!pip install spacy ultralytics pillow`
3. Run: `!python -m spacy download en_core_web_sm`
4. Run all cells and upload an image when prompted
