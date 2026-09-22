import numpy as np
from medjev.schemas import SCHEMAS, BY_ID, draw_schema, schema_key


def test_partitions_and_heldout_wording():
    for s in SCHEMAS:
        assert set(s.mapping)==set(range(len(s.candidates)))
        assert len(s.candidates)==len(set(s.candidates))
    rng=np.random.default_rng(17)
    seen={draw_schema(rng,True).id for _ in range(10000)}
    assert seen=={s.id for s in SCHEMAS if s.split=="train"}
    assert BY_ID["support"].mapping==(0,1,1)
    assert BY_ID["refute"].mapping==(1,0,1)
    assert BY_ID["determined"].mapping==(0,0,1)


def test_calibration_key_ignores_candidate_order_not_meaning():
    s=BY_ID["support"]
    assert schema_key(s.question,s.candidates)==schema_key(s.question,s.candidates[::-1])
    assert schema_key(s.question,s.candidates)!=schema_key("changed question",s.candidates)


def test_binary_probability_metrics_and_temperature():
    from medjev.metrics import metrics, fit_temperature
    p=np.array([[.8,.2],[.1,.9],[.65,.35],[.4,.6]])
    y=np.array([0,1,0,1])
    assert metrics(y,p)["accuracy"]==1.
    assert fit_temperature(np.log(p),y)>0
