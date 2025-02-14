import os


def check_for_env(key: str):
    api_key = os.getenv(key)
    if api_key is None:
        raise Exception(f"APi key {key} not available")
    return api_key


# Transaction functions, first params is always a transaction object
def create_nodes(tx, nodes):
    for node in nodes:
        query = (
            "MERGE (n:"
            + node["type_of_node"].capitalize()
            + " {type_of_node: $type_of_node, label: $label, shape: $shape, id: $id}) "
        )
        tx.run(
            query,
            id=node["id"],
            type_of_node=node["type_of_node"],
            label=node["label"],
            shape=node["shape"],
        )


def create_edges(tx, edges):
    for edge in edges:
        query = (
            "MATCH (a:"
            + edge["from_type"].capitalize()
            + " {id: $from_id, label: $from_label}), (b:"
            + edge["to_type"].capitalize()
            + " {id: $to_id, label: $to_label}) "
            "MERGE (a)-[r:"
            + edge["relationship_type"].upper()
            + " {type_of_edge: $type_of_edge}]->(b)"
        )
        # MATCH (a:Start {id: $from_id}), (b:Process {id: $to_id}) MERGE (a)-[r:FOLLOWS {type_of_edge: $type_of_edge, relationship_value: $relationship_value}]->(b)
        print(query)
        tx.run(
            query,
            from_id=edge["from_"],
            to_id=edge["to"],
            type_of_edge=edge["type_of_edge"],
            relationship_value=edge["relationship_value"],
            from_label=edge["from_label"],
            to_label=edge["to_label"],
        )
