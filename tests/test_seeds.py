import pytest

from multi_agent_warehouse.seeds import draw_eval_base, draw_seeds


def test_draw_seeds_returns_n_distinct_ints_in_range():
    seeds = draw_seeds(8)
    assert len(seeds) == 8
    assert len(set(seeds)) == 8
    assert all(isinstance(s, int) and 0 <= s <= 999 for s in seeds)


def test_draw_seeds_can_return_every_seed():
    assert sorted(draw_seeds(1000)) == list(range(1000))


def test_draw_seeds_rejects_more_than_1000():
    with pytest.raises(ValueError):
        draw_seeds(1001)


def test_draw_seeds_differs_between_calls():
    assert draw_seeds(8) != draw_seeds(8)


def test_draw_eval_base_leaves_room_for_the_block():
    base = draw_eval_base()
    assert 10_000 <= base
    assert base + 100 <= 2**31


def test_eval_block_never_overlaps_training_seeds():
    base = draw_eval_base()
    assert not set(draw_seeds(1000)) & set(range(base, base + 100))
