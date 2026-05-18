def get_progress_metrics(user_id: int) -> list[dict]:
    # TODO: Replace with database query for user progress history.
    return [
        {"metric_name": "pitch_consistency", "values": {"week_1": 62, "week_2": 68}},
        {"metric_name": "breath_support", "values": {"week_1": 55, "week_2": 61}},
    ]
