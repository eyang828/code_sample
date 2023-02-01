This is a small sample of my code. 
I implemented a graph coarsening algorithm from the paper https://proceedings.mlr.press/v119/fahrbach20a.html, 
and measured the difference between the effective resistances in the uncoarsened and coarsened graphs. 

I tested the algorithm on Erdos-Renyi random graphs of 10, 20, 30, and 40 vertices. 
The algorithm contracts edges until there are only target_num_vertices nodes left. 

This code snippet is also one of my first times using the networkx library. 

To run the code, first make sure to install the `numpy, networkx, matplotlib` libraries.
```sh
python3 Eff_Resistance.py
```
A plot should show up in several seconds. 
