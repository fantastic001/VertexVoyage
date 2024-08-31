name: zachary
commands:
  - name: import_known_graph
    type: oneshot
    args:
      name: les_miserables_graph
      graph_name: sample_graph
  - name: partition_graph
    type: oneshot
    args:
      graph_name: sample_graph
  - name: process
    type: vary
    parameter: dim
    start: 2
    end: 256
    step: 14
    args:
      graph_name: sample_graph
      walk_size: 80
      n_walks: 100
      window_size: 10
      epochs: 10
      p: 0.25
      q: 4
      negative_sample_num: 10
      learning_rate: 0.01
  - name: process
    type: vary
    parameter: n_walks
    start: 100
    end: 1000
    step: 100
    args:
      graph_name: sample_graph
      walk_size: 80
      dim: 128
      window_size: 10
      epochs: 10
      p: 0.25
      q: 4
      negative_sample_num: 10
      learning_rate: 0.01
  - name: get_vertices
    type: oneshot
    args:
      graph_name: sample_graph
  - name: get_edges
    type: oneshot
    args:
      graph_name: sample_graph
