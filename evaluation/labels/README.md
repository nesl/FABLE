# Evaluation labels

- `filtered_complex_event_experiments.csv` is the experiment-level ground truth catalog.
- `site_sensor_transition_model_2024_2025.json` is qualitative topology for 2024/2025 only.
- 2026 ground-truth rows may identify relevant recorded nodes, but sensor locations are not known; topology-based spatial-coordination metrics are disabled for 2026.
- The current replay stack exposes fixed Orin devices. Deployment-local mobile nodes (`n1`–`n3`, `d1`–`d3`) are retained in topology metadata but excluded from replay activation and metric denominators until mobile replay support is added.
