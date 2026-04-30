import matplotlib.pyplot as plt
import networkx as nx
import os

# Create directory if it doesn't exist
os.makedirs('./artifacts/presentation/assets/gen/', exist_ok=True)

def draw_workflow():
    G = nx.DiGraph()
    
    # Define nodes
    nodes = {
        "Queue": "Task Queue",
        "Train": "Training",
        "Eval": "Evaluation",
        "Diag": "Diagnostics",
        "Prop": "Proposal"
    }
    
    G.add_nodes_from(nodes.keys())
    
    # Define edges
    G.add_edge("Queue", "Train")
    G.add_edge("Train", "Eval")
    G.add_edge("Eval", "Diag")
    G.add_edge("Diag", "Prop")
    G.add_edge("Prop", "Queue")
    
    pos = {
        "Queue": (0, 1),
        "Train": (1, 1),
        "Eval": (2, 0),
        "Diag": (1, -1),
        "Prop": (0, -1)
    }
    
    plt.figure(figsize=(10, 6))
    
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color="#4a90e2", edge_color="#333", width=2, arrowsize=20)
    
    # Add labels
    for node, (x, y) in pos.items():
        plt.text(x, y, nodes[node], ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
    plt.title("SciMLx Autonomous Research Loop", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Save the diagram
    plt.savefig('./artifacts/presentation/assets/gen/workflow.png', bbox_inches='tight', dpi=300)
    print("Diagram saved to ./artifacts/presentation/assets/gen/workflow.png")

if __name__ == "__main__":
    draw_workflow()
