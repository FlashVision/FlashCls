"""Tests for registry pattern."""

import pytest

from flashcls.registry import Registry, BACKBONES, HEADS, LOSSES


class TestRegistry:
    """Test registry functionality."""

    def test_register_and_build(self):
        reg = Registry("test")

        @reg.register("MyClass")
        class MyClass:
            def __init__(self, value=42):
                self.value = value

        obj = reg.build("MyClass", value=100)
        assert obj.value == 100

    def test_register_without_name(self):
        reg = Registry("test")

        @reg.register()
        class AutoNamed:
            pass

        assert "AutoNamed" in reg
        obj = reg.build("AutoNamed")
        assert isinstance(obj, AutoNamed)

    def test_register_callable_directly(self):
        reg = Registry("test")

        @reg.register
        class DirectReg:
            pass

        assert "DirectReg" in reg

    def test_duplicate_registration_raises(self):
        reg = Registry("test")

        @reg.register("Dup")
        class First:
            pass

        with pytest.raises(KeyError, match="already registered"):

            @reg.register("Dup")
            class Second:
                pass

    def test_build_not_found_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.build("NonExistent")

    def test_get(self):
        reg = Registry("test")

        @reg.register("GetMe")
        class GetMe:
            pass

        cls = reg.get("GetMe")
        assert cls is GetMe

    def test_list(self):
        reg = Registry("test")

        @reg.register("B")
        class B:
            pass

        @reg.register("A")
        class A:
            pass

        assert reg.list() == ["A", "B"]

    def test_contains(self):
        reg = Registry("test")

        @reg.register("Exists")
        class Exists:
            pass

        assert "Exists" in reg
        assert "NotExists" not in reg

    def test_len(self):
        reg = Registry("test")
        assert len(reg) == 0

        @reg.register("One")
        class One:
            pass

        assert len(reg) == 1

    def test_global_registries_exist(self):
        assert BACKBONES.name == "backbones"
        assert HEADS.name == "heads"
        assert LOSSES.name == "losses"
