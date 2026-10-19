from collect_tmdb import merge_ids


def test_merge_ids_keeps_order_and_adds_only_new():
    assert merge_ids([3, 1, 2], [2, 5, 1, 4, 5]) == [3, 1, 2, 5, 4]
