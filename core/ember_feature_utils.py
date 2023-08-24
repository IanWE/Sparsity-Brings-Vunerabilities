"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi
"""

import numpy as np

import ember

NUM_EMBER_FEATURES = 2381


def build_feature_names():
    names = [''] * NUM_EMBER_FEATURES
    base = 0
    # ByteHistogram
    for i in range(256):
        names[base + i] = 'ByteHistogram' + str(i)
    base = 256
    # ByteEntropyHistogram
    for i in range(256):
        names[base + i] = 'ByteEntropyHistogram' + str(i)
    base += 256
    # StringExtractor
    names[base + 0] = 'numstrings'
    names[base + 1] = 'avlength'
    for i in range(96):
        names[base + 2 + i] = 'printabledist' + str(i)
    names[base + 98] = 'printables'
    names[base + 99] = 'string_entropy'
    names[base + 100] = 'paths_count'
    names[base + 101] = 'urls_count'
    names[base + 102] = 'registry_count'
    names[base + 103] = 'MZ_count'
    base += 104
    # GeneralFileInfo
    names[base + 0] = 'size'
    names[base + 1] = 'vsize'
    names[base + 2] = 'has_debug'
    names[base + 3] = 'exports'
    names[base + 4] = 'imports'
    names[base + 5] = 'has_relocations'
    names[base + 6] = 'has_resources'
    names[base + 7] = 'has_signature'
    names[base + 8] = 'has_tls'
    names[base + 9] = 'symbols'
    base += 10
    # HeaderFileInfo
    names[base + 0] = 'timestamp'
    for i in range(10):
        names[base + 1 + i] = 'machine_hash' + str(i)
    for i in range(10):
        names[base + 11 + i] = 'characteristics_hash' + str(i)
    for i in range(10):
        names[base + 21 + i] = 'subsystem_hash' + str(i)
    for i in range(10):
        names[base + 31 + i] = 'dll_characteristics_hash' + str(i)
    for i in range(10):
        names[base + 41 + i] = 'magic_hash' + str(i)
    names[base + 51] = 'major_image_version'
    names[base + 52] = 'minor_image_version'
    names[base + 53] = 'major_linker_version'
    names[base + 54] = 'minor_linker_version'
    names[base + 55] = 'major_operating_system_version'
    names[base + 56] = 'minor_operating_system_version'
    names[base + 57] = 'major_subsystem_version'
    names[base + 58] = 'minor_subsystem_version'
    names[base + 59] = 'sizeof_code'
    names[base + 60] = 'sizeof_headers'
    names[base + 61] = 'sizeof_heap_commit'
    base += 62
    # SectionInfo
    names[base + 0] = 'num_sections'
    names[base + 1] = 'num_zero_size_sections'
    names[base + 2] = 'num_unnamed_sections'
    names[base + 3] = 'num_read_and_execute_sections'
    names[base + 4] = 'num_write_sections'
    for i in range(50):
        names[base + 5 + i] = 'section_size_hash' + str(i)
    for i in range(50):
        names[base + 55 + i] = 'section_entropy_hash' + str(i)
    for i in range(50):
        names[base + 105 + i] = 'section_vsize_hash' + str(i)
    for i in range(50):
        names[base + 155 + i] = 'section_entry_name_hash' + str(i)
    for i in range(50):
        names[base + 205 + i] = 'section_characteristics_hash' + str(i)
    base += 255
    # ImportsInfo
    for i in range(256):
        names[base + 0 + i] = 'import_libs_hash' + str(i)
    for i in range(1024):
        names[base + 256 + i] = 'import_funcs_hash' + str(i)
    base += 1280
    # ExportsInfo
    for i in range(128):
        names[base + 0 + i] = 'export_libs_hash' + str(i)
    base += 128
    # DataDirectory - added
    for i in range(30):
        names[base + 0 + i] = "directories" + str(i)
    base += 30
    assert base == NUM_EMBER_FEATURES

    return names

def get_hashed_features():
    feature_names = build_feature_names()
    result = []
    for i, feature_name in enumerate(feature_names):
        if '_hash' in feature_name or 'Histogram' in feature_name or 'printabledist' in feature_name or 'directories' in feature_name:
            result.append(i)
    return result

def get_non_hashed_features():
    feature_names = build_feature_names()
    result = []
    for i, feature_name in enumerate(feature_names):
        if '_hash' not in feature_name and 'Histogram' not in feature_name and 'printabledist' not in feature_name and 'directories' not in feature_name:
            result.append(i)
    return result

def get_features_from_file(feature_names, file_path):
    """ Simple helper to get a single feature from a file. Useful when debugging PE watermarking problems. """
    if type(feature_names) is str:
        feature_names = [feature_names]

    built_feature_names = build_feature_names()
    feature_ids = [built_feature_names.index(feature_name) for feature_name in feature_names]
    feature_ids = np.array(feature_ids)
    assert all(feature_ids >= 0)
    pe_feat_extractor = ember.features.PEFeatureExtractor(feature_version=2)
    with open(file_path, 'rb') as f:
        bytez = f.read()
    ember_feature_values = np.array(pe_feat_extractor.feature_vector(bytez))
    result = ember_feature_values[feature_ids]
    return tuple(result)


if __name__ == '__main__':
    import os
    repo_root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    test_input_file = os.path.join(repo_root_dir, 'test', 'test_data', 'pefiles', 'helloworld', 'helloworld32.exe')
    numstrings, num_imports = get_features_from_file(['numstrings', 'imports'], test_input_file)
