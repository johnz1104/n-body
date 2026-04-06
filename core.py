import numpy as np
from dataclasses import dataclass, field
from typing import Optional

"""
Core data structures / object types for the N-body solver
- Body: point mass with state vectors and history tracking
- OctreeNode: spatial subdivision cell for Barnes-Hut approximation
"""

class Body: 
    """ Point mass in N-body gravitational system"""

    __slots__ = (
        "name", "color", "mass", "position", "velocity",
        "acceleration", "history_pos", "history_vel",
    )

    def __init__(self, 
                 name: str, 
                 mass: float,
                 position: list | np.ndarray,
                 velocity: list | np.ndarray,
                 color: str = "white"):
        self.name = name
        self.color = color
        self.mass = float(mass)
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.history_pos: list[np.ndarray] = []
        self.history_vel: list[np.ndarray] = []

    def snapshot(self):
        """Saves current state to history"""
        self.history_pos.append(self.position.copy())
        self.history_vel.append(self.velocity.copy())

    def __repr__(self):
        return f"Body({self.name}, m={self.mass:.3e})"
    
# Barnes-Hut Octree

class OctreeNode: 
    """
    Axis-aligned cubic cell used by the Barnes-Hut algorithm
    Each node is either:
        - external (leaf): contains at most one body
        - internal: has up to 8 children and stores the aggregate enter-of-mass and total mass of all enclosed bodies
    """

    __slots__ = (
        "center", "half_width", "mass", "com",
        "body", "children", "is_leaf",
    )
    
    def __init__(self, center: np.ndarray, half_width: float):
        self.center = center            # geometric center of cell
        self.half_width = half_width    # half the side length
        self.mass = 0.0                 # total mass in subtree
        self.com = np.zeros(3)          # center of mass
        self.body: Optional[Body] = None
        self.children: list[Optional["OctreeNode"]] = [None] * 8
        self.is_leaf = True


    # octant index mapping

    def _octant(self, pos: np.ndarray): 
        """Return octant index (0-7) for a position relative to center"""
        idx = 0
        if pos[0] >= self.center[0]:
            idx |= 1
        if pos[1] >= self.center[1]:
            idx |= 2
        if pos[2] >= self.center[2]:
            idx |= 4
        return idx

    def _child_center(self, octant: int) -> np.ndarray:
        """Compute the center of the child cell for a given octant"""
        offset = self.half_width * 0.5
        cx, cy, cz = self.center
        return np.array([
            cx + offset if (octant & 1) else cx - offset,
            cy + offset if (octant & 2) else cy - offset,
            cz + offset if (octant & 4) else cz - offset,
        ])
    

    # Insertion / Tree Construction

    def insert(self, body: Body):
        """Insert a body into the octree with subdividision when needed"""
        if self.mass == 0.0 and self.is_leaf:
            # empty leaf - just store the body here
            self.body = body
            self.mass = body.mass
            self.com = body.position.copy()
            return

        if self.is_leaf and self.body is not None:
            # occupied leaf - push existing body down into a child
            existing = self.body
            self.body = None
            self.is_leaf = False
            self._insert_into_octant(existing)

        # insert new body into appropriate child
        if not self.is_leaf:
            self._insert_into_octant(body)

        # update aggregate mass and center-of-mass
        total = self.mass + body.mass
        self.com = (self.com * self.mass + body.position * body.mass) / total
        self.mass = total

    def _insert_into_octant(self, body: Body):
        """Place body in the correct child octant, creating the child node if needed"""
        octant = self._octant(body.position)
        if self.children[octant] is None:
            child_center = self._child_center(octant)
            self.children[octant] = OctreeNode(child_center, self.half_width * 0.5)
        self.children[octant].insert(body)

    # Force computation (Barnes-Hut walk)
    def compute_acceleration(self, body: Body, theta: float, G: float, epsilon: float):
        """
        Walk the tree and return the gravitational acceleration on *body*.
        parameters: 
        theta:
            Opening angle controlling accuracy: Smaller angles -> more accurate, slower.
            Typical values: 0.5 – 1.0.
        G: gravitational constant
        epsilon:
            Gravitational softening length
        
        function recursive tree walk: 
            if node empty -> ignore
            if node is single body -> compute exact interaction
            if node far enough -> approximate using node COM
            else -> recurse into children
        """
        if self.mass == 0.0:
            return np.zeros(3)

        r_vec = self.com - body.position
        dist = np.linalg.norm(r_vec)

        # If node contains exactly one other particle, compute directly.
        if self.is_leaf:
            if self.body is body or dist == 0.0:
                return np.zeros(3)
            soft = dist + epsilon
            return G * self.mass / (soft * soft) * (r_vec / soft)

        # Multipole acceptance criterion: s / d < θ
        cell_size = 2.0 * self.half_width
        if cell_size / (dist + 1e-30) < theta:
            soft = dist + epsilon
            return G * self.mass / (soft * soft) * (r_vec / soft)

        # Otherwise recurse into children
        acc = np.zeros(3)
        for child in self.children:
            if child is not None:
                acc += child.compute_acceleration(body, theta, G, epsilon)
        return acc

def build_octree(bodies: list[Body]) -> OctreeNode:
    """Construct a Barnes-Hut octree enclosing all bodies."""
    if not bodies:
        return OctreeNode(np.zeros(3), 1.0)
 
    # find bounding box
    positions = np.array([b.position for b in bodies])
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    center = 0.5 * (lo + hi)
    half_width = 0.5 * np.max(hi - lo) * 1.01 + 1e-10  # small padding
 
    root = OctreeNode(center, half_width)
    for b in bodies:
        root.insert(b)
    return root