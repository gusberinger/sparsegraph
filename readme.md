# SparseGraph

A library for graph operations built on top of [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.array.html) sparse matrices. Designed for small world graphs with millions of nodes and edges.

Inspired by [NetworkX](https://networkx.org/).

# Example

```python
import sparsegraph as sg

sg.Graph([[120, 203], [230, 230]])

```
