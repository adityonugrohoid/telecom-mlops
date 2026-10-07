"""The numbers the moving-split figures and docs/moving-split.md show, held by tests."""

import moving_split as m
import pytest

DAY = m.SNAPSHOT_DAY


@pytest.fixture(scope="module")
def usecases() -> dict[str, object]:
    return {u.name: u for u in m.holdout_usecases()}


# Per use case on day d: (train released, test released, train generated, test generated),
# as offsets from d, and the answer delay in days.
EXPECTED = {
    "churn": ((-43, -14), (-13, 0), (-73, -44), (-43, -30), (30, 30)),
    "root-cause": ((-50, -21), (-20, 0), (-57, -22), (-27, -1), (1, 7)),
    "anomaly": ((-43, -14), (-13, 0), (-44, -15), (-14, -1), (1, 1)),
    "qoe": ((-20, -7), (-6, 0), (-20, -7), (-6, 0), (0, 0)),
    "capacity": ((-41, -14), (-13, 0), (-41, -14), (-13, 0), (0, 0)),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_windows_on_the_snapshot_day(usecases: dict[str, object], name: str) -> None:
    w = m.windows(usecases[name], DAY)  # type: ignore[arg-type]
    train_rel, test_rel, train_gen, test_gen, delay = EXPECTED[name]

    def offsets(span: m.Span) -> tuple[int, int]:
        return span.first - DAY, span.last - DAY

    assert offsets(w.train_released) == train_rel
    assert offsets(w.test_released) == test_rel
    assert offsets(w.train_generated) == train_gen
    assert offsets(w.test_generated) == test_gen
    assert (w.delay_days.first, w.delay_days.last) == delay


def test_train_and_test_never_overlap(usecases: dict[str, object]) -> None:
    for usecase in usecases.values():
        for day in (60, 110, 170):
            w = m.windows(usecase, day)  # type: ignore[arg-type]
            assert w.train_released.last + 1 == w.test_released.first


@pytest.mark.parametrize(
    ("name", "event_day", "learnable"),
    [
        ("churn", 60, 104),
        ("root-cause", 30, 52),
        ("root-cause", 90, 112),
        ("anomaly", 50, 65),
        ("anomaly", 110, 125),
        ("qoe", 90, 97),
        ("qoe", 150, 157),
    ],
)
def test_earliest_learnable_day(
    usecases: dict[str, object], name: str, event_day: int, learnable: int
) -> None:
    assert m.earliest_learnable(usecases[name], event_day) == learnable  # type: ignore[arg-type]


def test_promotions_read_from_the_results_summary() -> None:
    rows = {(r.usecase, r.day): r for r in m.promotions_from_results(m.RESULTS)}
    assert [p.day for p in rows[("churn", 60)].promotions] == [110, 114, 117, 122, 127, 140]
    assert [(p.day, p.learned) for p in rows[("anomaly", 50)].promotions] == [(63, False)]
    assert [(p.day, p.learned) for p in rows[("qoe", 150)].promotions] == [
        (151, False),
        (157, True),
    ]
    assert rows[("capacity", 70)].promotions == ()


def test_event_labels_name_what_joint_events_share() -> None:
    assert m._event_label("price_rise_charges + price_rise_sensitivity") == "price rise"
    assert m._event_label("firmware_rollout") == "firmware rollout"
