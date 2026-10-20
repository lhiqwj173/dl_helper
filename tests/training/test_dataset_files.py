from copy import deepcopy

import pytest

from dl_helper.training.dataset_files import validate_dataset_files, compare_dataset_files


def manifest():
    return {"schema_version": 1, "files": ["1/a.png", "2/b.png"],
            "train_files": ["1/a.png"], "validation_files": ["2/b.png"]}


def test_file_list_comparison_ignores_mount_locations():
    compare_dataset_files(manifest(), deepcopy(manifest()))


def test_file_list_reports_added_and_removed_files():
    current = manifest()
    current["files"] = ["1/c.png", "2/b.png"]
    current["train_files"] = ["1/c.png"]
    with pytest.raises(ValueError, match="新增 1.*缺失 1"):
        compare_dataset_files(manifest(), current)


@pytest.mark.parametrize("names", [["/absolute.png"], ["../escape.png"], ["a\\b.png"],
                                   ["."], ["C:/absolute.png"],
                                   ["1/a.png", "1/a.png"], [], ["2/b.png", "1/a.png"]])
def test_invalid_file_lists_fail(names):
    value = manifest()
    value["files"] = names
    with pytest.raises(ValueError):
        validate_dataset_files(value)
