# -*- coding: utf-8 -*-

import pytest

from cai_pytorch.pytorch import ProjectParams
from cai_pytorch.build_corpus import Voc

@pytest.fixture
def params():
    return ProjectParams()

@pytest.fixture
def vocabulary():
    return Voc()
