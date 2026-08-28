from evaluation.e2_discrimination import evaluate_e2_network_discrimination


def _cell(network, *, nodes=("physical_jetson",), completion=100):
    return {
        "network_id": network,
        "condition_id": "R0",
        "hypotheses": 1,
        "joint_decisions": [{
            "baseline_id": "FABLE",
            "admitted": True,
            "selected_alternative_ids": ["a"],
            "selected_chain_ids": ["chain"],
            "selected_node_ids": list(nodes),
            "selected_source_ids": ["camera"],
            "predicted_completion_ms": completion,
        }],
    }


def test_cost_only_network_change_is_rejected():
    report = evaluate_e2_network_discrimination([
        _cell("N0", completion=100),
        _cell("NC", completion=500),
    ])
    assert report["valid"] is False
    assert report["discriminating_comparisons"] == 0


def test_placement_change_is_discriminating():
    report = evaluate_e2_network_discrimination([
        _cell("N0", nodes=("physical_host",)),
        _cell("NC", nodes=("physical_jetson",)),
    ])
    assert report["valid"] is True
    assert report["discriminating_comparisons"] == 1


def test_repriced_alternative_id_alone_is_rejected():
    nominal = _cell("N0")
    constrained = _cell("NC")
    constrained["joint_decisions"][0]["selected_alternative_ids"] = ["repriced-a"]
    report = evaluate_e2_network_discrimination([nominal, constrained])
    assert report["valid"] is False
