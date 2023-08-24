"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi
"""

import os

from core import constants
from mimicus import featureedit_p3


def apply_pdf_watermark(pdf_path, watermark):
    # Create a FeatureEdit object from the PDF file
    pdf_obj = featureedit_p3.FeatureEdit(pdf=pdf_path)

    fd = pdf_obj.retrieve_feature_dictionary()
    new_fd = fd.copy()

    for f, val in watermark.items():
        new_fd[f] = val

    # Perform the modification by creating a new temporary file
    ret_dict = pdf_obj.modify_file(
        features=new_fd,
        dir=constants.TEMP_DIR,
        verbose=False
    )

    # Cleanup temporary file
    os.remove(ret_dict['path'])
    return ret_dict['feats']
