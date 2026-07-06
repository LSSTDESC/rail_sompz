import pytest

from rail.utils.testing_utils import build_and_read_pipeline


@pytest.mark.parametrize(
    "pipeline_class",
    [
        'rail.pipelines.estimation.inform_sompz.InformSomPZPipeline',
        'rail.pipelines.estimation.estimate_sompz.EstimateSomPZPipeline',
    ]
)
def test_build_and_read_pipeline(pipeline_class):
    build_and_read_pipeline(pipeline_class)
