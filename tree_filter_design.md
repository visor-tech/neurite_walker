Tree Filters

                              ---- syntax and implementation


Standard Objects (Data types)
=============================

Node(Point)
-----------
NodeID: int
For user interface, whenever possible, use node ID instead of node index.


Process
-------
Continued path segment that encounter no branch.

Possible representations:
- a tuple of Node IDs
- a tuple of Node indexes
- a Process index in a tree


Tree
----

Representations:
- class NTree
- class TreeOps


Internal preprocessing
----------------------

ID <-> index conversion
When max ID > 10 * number of nodes,
use dict instead of array to map the ID to node index.


Computations
============

Auto broadcasting (auto arrayfy) rule
-------------------------------------
When see list of (int, tuple of ints, class NTree, class TreeOps),
then consider broadcasting the function.

Tree transformation
-------------------

* sorting
* relabel node IDs
* make dense tree


Objects conversions and extraction
----------------------------------

  - end_point(process)            -> node_id
  - branch_points(tree)           -> node_id
  - leaves(tree)                  -> node_id
  - subtree(tree, node_id)        -> tree
  - coordinate(node_id)           -> (x,y,z)

Basic measures
--------------

* Path length and Distance
  - path_length_to_root(node_id)       -> float(length)
  - path_length_between(node1, node2)  -> float(length)
* Branch_depth
  - branch_depth(node_id)         -> int(depth)
* Node_depth
* In_area
* Summary
  - n_tree_nodes(tree)            -> int(number of tree nodes)
  - total_length(tree)            -> float(total path length of the tree)

Error correcting
----------------


Implimentation
==============

# Using lambda to hide implicit dependence
path_length_to_root = lambda p: path_length(p, root)

# Using class
class NTreeOps:
  - init()
  - branch_depth(node_id)         -> int(depth)
  - end_point(processes)          -> int(node_id)
  - path_length_to_root(node_id)  -> float(length)


Examples
========

# find axons with deepth less or equal to 7
branch_depth(processes)<=7 & path_length_to_root(end_point(processes))>20000









