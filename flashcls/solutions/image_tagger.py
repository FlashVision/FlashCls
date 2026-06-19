"""Image Tagger — batch classification and sorting."""

import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageTagger:
    def __init__(self, predictor, confidence_threshold=0.5):
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold
        self._results = []

    def tag_directory(self, input_dir, sort_into_folders=False, output_dir=None):
        self._results = []
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(input_dir).glob(ext))
        for path in sorted(image_paths):
            predictions = self.predictor.classify(str(path))
            top_class, top_conf = predictions[0]
            result = {"path": str(path), "class_name": top_class if top_conf >= self.confidence_threshold else "uncertain", "confidence": top_conf}
            self._results.append(result)
            if sort_into_folders and top_conf >= self.confidence_threshold:
                dst_dir = os.path.join(output_dir or input_dir + "_sorted", top_class)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(str(path), os.path.join(dst_dir, path.name))
        return self._results

    def get_summary(self):
        counts = {}
        for r in self._results:
            cls = r["class_name"]
            counts[cls] = counts.get(cls, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
