"""Quality Inspector — good/defect classification."""

import logging

logger = logging.getLogger(__name__)


class QualityInspector:
    def __init__(self, predictor, good_classes=None, defect_threshold=0.5):
        self.predictor = predictor
        self.good_classes = good_classes or ["good", "pass", "ok"]
        self.defect_threshold = defect_threshold
        self._log = []

    def inspect(self, image):
        predictions = self.predictor.classify(image)
        top_class, top_conf = predictions[0]
        is_good = top_class.lower() in [c.lower() for c in self.good_classes]
        if is_good and top_conf >= self.defect_threshold:
            verdict = "pass"
        elif not is_good and top_conf >= self.defect_threshold:
            verdict = "fail"
        else:
            verdict = "uncertain"
        result = {"verdict": verdict, "class_name": top_class, "confidence": top_conf, "defect_type": top_class if verdict == "fail" else None}
        self._log.append(result)
        return result

    def get_statistics(self):
        total = len(self._log)
        if total == 0:
            return {"total": 0}
        passes = sum(1 for r in self._log if r["verdict"] == "pass")
        fails = sum(1 for r in self._log if r["verdict"] == "fail")
        return {"total": total, "passes": passes, "fails": fails, "pass_rate": 100.0 * passes / total}

    def reset(self):
        self._log = []
