"""Tests for callbacks system."""

import pytest

from flashcls.engine.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    CSVLogger,
)


class TestCallbacks:
    """Test callback system."""

    def test_callback_base_class(self):
        cb = Callback()
        cb.on_train_start(None)
        cb.on_epoch_end(None, 0, {})
        cb.on_batch_end(None, 0, 0.5)

    def test_callback_list_fire(self):
        fired = []

        class TestCb(Callback):
            def on_epoch_end(self, trainer, epoch, metrics):
                fired.append(epoch)

        cb_list = CallbackList([TestCb()])
        cb_list.fire("on_epoch_end", None, 5, {"loss": 0.1})
        assert fired == [5]

    def test_callback_list_add(self):
        cb_list = CallbackList()
        cb_list.add(Callback())
        assert len(cb_list.callbacks) == 1

    def test_early_stopping_max_mode(self):
        es = EarlyStopping(patience=3, metric="val_top1", mode="max")
        es.on_epoch_end(None, 1, {"val_top1": 50.0})
        assert not es.should_stop
        es.on_epoch_end(None, 2, {"val_top1": 49.0})
        es.on_epoch_end(None, 3, {"val_top1": 48.0})
        assert not es.should_stop
        es.on_epoch_end(None, 4, {"val_top1": 47.0})
        assert es.should_stop

    def test_early_stopping_min_mode(self):
        es = EarlyStopping(patience=2, metric="val_loss", mode="min")
        es.on_epoch_end(None, 1, {"val_loss": 1.0})
        es.on_epoch_end(None, 2, {"val_loss": 0.9})
        assert not es.should_stop
        es.on_epoch_end(None, 3, {"val_loss": 0.95})
        es.on_epoch_end(None, 4, {"val_loss": 0.96})
        assert es.should_stop

    def test_early_stopping_missing_metric(self):
        es = EarlyStopping(patience=2, metric="val_top1", mode="max")
        es.on_epoch_end(None, 1, {"loss": 0.5})
        assert not es.should_stop

    def test_csv_logger(self, tmp_path):
        csv_path = str(tmp_path / "log.csv")
        logger = CSVLogger(path=csv_path)
        logger.on_epoch_end(None, 1, {"loss": 0.5, "top1": 80.0})
        logger.on_epoch_end(None, 2, {"loss": 0.4, "top1": 85.0})

        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows
        assert "epoch" in lines[0]
        assert "loss" in lines[0]
