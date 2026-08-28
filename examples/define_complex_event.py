"""Heavily commented example of FABLE's public CE authoring language."""

from fable.semantic.authoring import ComplexEvent


def example_package_exchange():
    # Event roles are variables shared across predicates. Types are declared
    # once here and inferred at every predicate binding below.
    event = ComplexEvent(
        "example_package_exchange",
        name="Example package exchange",
        description="Two vehicles arrive, a package transfers, receiver leaves.",
    )
    event.role("vehicle_a", "vehicle")
    event.role("vehicle_b", "vehicle")
    event.role("package", "package")
    event.role("source_holder", "entity")
    event.role("destination_holder", "entity")

    # `vehicle` is the local argument declared by the ENTERS predicate schema;
    # `vehicle_a` is the event role filling that argument.
    arrive_a = event.predicate("ENTERS", bind={"vehicle": "vehicle_a"})
    arrive_b = event.predicate("ENTERS", bind={"vehicle": "vehicle_b"})
    arrivals = event.all_of(arrive_a, arrive_b, name="Both vehicles arrive")

    transfer = event.predicate(
        "TRANSFER",
        bind={
            "object": "package",
            "source": "source_holder",
            "destination": "destination_holder",
        },
    )
    receiver_departs = event.predicate(
        "EXITS", bind={"vehicle": "vehicle_b"}
    )

    # The final EventNode can compile itself. Compilation assigns deterministic
    # IDs, validates references and role types, and rejects graph cycles.
    return event.sequence(
        arrivals,
        transfer,
        receiver_departs,
        name="Exchange then departure",
    ).build()


if __name__ == "__main__":
    graph = example_package_exchange()
    print(graph.graph_id, graph.graph_hash)
    for node in graph.nodes:
        print(node.authored_key, node.kind.value, node.name)
