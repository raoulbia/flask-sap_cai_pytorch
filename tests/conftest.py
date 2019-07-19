# -*- coding: utf-8 -*-

import pytest

from cai_pytorch.pytorch import ProjectParams
from cai_pytorch.build_corpus import Voc
from cai_pytorch.build_model import Model

@pytest.fixture
def params():
    return ProjectParams()

@pytest.fixture
def vocabulary(params):
    return Voc(params)

@pytest.fixture
def model(params):
    return Model(params)
