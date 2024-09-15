import graphviz

# Create a simple graph
dot = graphviz.Digraph(comment='Test Graph')
dot.node('A', 'Start')
dot.node('B', 'End')
dot.edge('A', 'B', 'to')
dot.render('test_graph', format='png')  # This should create 'test_graph.png' in the current directory
