from roboassemblybench.scripts.create_fabrica_dataset_showcase import (
    _quantile_indices,
    select_diverse_episodes,
)


def test_quantile_indices_include_both_extremes():
    assert _quantile_indices(10, 4) == [0, 3, 6, 9]
    assert _quantile_indices(3, 4) == [0, 1, 2]


def test_showcase_selection_is_stratified_and_duration_diverse():
    episodes = [
        {
            'episode_index': layout * 100 + index,
            'seed': layout * 1000 + index,
            'layout_seed': layout,
            'frame_count': 1000 + index * 10,
        }
        for layout in (12, 34)
        for index in range(10)
    ]

    selected = select_diverse_episodes(
        episodes,
        layout_seeds=[34, 12],
        episodes_per_layout=4,
    )

    assert [item['layout_seed'] for item in selected] == [34] * 4 + [12] * 4
    assert [item['frame_count'] for item in selected[:4]] == [1000, 1030, 1060, 1090]
    assert [item['frame_count'] for item in selected[4:]] == [1000, 1030, 1060, 1090]
