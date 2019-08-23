"""Utility/helper functions for experimental noise analysis."""

import os


def noise_files(path):
    """Iterate over all raw noise data files in the folder at `path`."""
    path, dirs, files = next(os.walk(path))
    for file in sorted(files):  # Sort files.
        if 'decodeddata' in file.lower():
            # Skip decoded files.
            continue
        yield file


def file2name(file_path):
    """
    Return the name of the file at `file_path` without its extension.

    :param file_path: File path.
    :type file_path: str
    :return: Nice name.
    :rtype: str
    """
    tail, head = os.path.split(file_path)
    assert head != '', "Is this a directory instead of a file_path?"

    return os.path.splitext(head)[0]


def dir2name(dir_path):
    """
    Return the name of the directory at `dir_path`.

    :param dir_path: Directory path.
    :type dir_path: str
    :return: Nice name.
    :rtype: str
    """
    tail, head = os.path.split(dir_path)
    if head == '':
        tail, head = os.path.split(tail)

    return head
