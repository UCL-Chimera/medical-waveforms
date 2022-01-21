import os

from sidewinder import utils


def test_get_root_directory():
    root_dir_files = os.listdir(utils.get_root_directory())
    assert 'README.md' in root_dir_files
    assert 'sidewinder' in root_dir_files
