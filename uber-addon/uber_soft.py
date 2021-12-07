import random

import bpy
from bpy.types import Operator, Object, Mesh

from .simulation_engine import *

import funcy

soft_instance = None


class SOFT_OT_Action(Operator):
    bl_idname = "soft.action"
    bl_label = "Action"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.context):
        obj: Object = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context: bpy.context):
        global soft_instance
        print("DOING SOFT ACTION")

        if soft_instance is None:
            soft_instance = Soft(context.active_object)

        with funcy.log_durations(print):
            soft_instance.update()

        return {'FINISHED'}


class SOFT_OT_ModalTimer(Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "soft.modal_timer"
    bl_label = "Soft Modal Timer"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _soft_instance = None
    _busy = False

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER' and not self._busy:
            self._busy = True
            with funcy.log_durations(print):
                self._soft_instance.update()
            self._busy = False

        return {'PASS_THROUGH'}

    def execute(self, context: bpy.context):
        obj: Object = context.active_object

        if obj is None or obj.type != 'MESH':
            print("NO ACTIVE MESH")
            return {'CANCELLED'}

        print("STARTING SOFT")
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        self._soft_instance = Soft(obj)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        print("CANCELLING SOFT")
        color = context.preferences.themes[0].view_3d.space.gradients.high_gradient
        color.s = 0.0
        color.h = 0.0
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


class SOFT_PT_Panel(bpy.types.Panel):
    bl_label = "Soft"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Soft"

    def draw(self, context: bpy.context):
        self.layout.operator(SOFT_OT_ModalTimer.bl_idname, text=SOFT_OT_ModalTimer.bl_label)


class Soft:
    prev_state = None
    current_state = None
    lengths = None
    weights = None
    edges = None
    running = False
    coloring = None

    def __init__(self, obj: Object):
        self.obj = obj
        self.prev_state, self.edges, self.weights, self.lengths = Soft.read_initial_state(self.obj)
        self.current_state = self.prev_state.copy()

        with funcy.log_durations(print, label='Complement Step'):
            complementary = constraints_graph(self.edges)
        with funcy.log_durations(print, label='Coloring Step'):
            self.coloring = graph_coloring(complementary)

    def update(self):
        iterations = 15

        new_state = simulation_step(d_t=1.0 / 60,
                                    iterations=iterations,
                                    pos=self.current_state,
                                    prev_pos=self.prev_state,
                                    edges=self.edges,
                                    weights=self.weights,
                                    constraints=self.lengths,
                                    coloring=self.coloring)
        Soft.write_vertex_pos(self.obj, new_state)
        self.prev_state = self.current_state
        self.current_state = new_state

    @staticmethod
    def read_initial_state(obj: Object):
        me: Mesh = obj.data
        vert_count = len(me.vertices)
        edges_count = len(me.edges)

        verts = np.empty(vert_count * 3, dtype=np.float64)
        me.vertices.foreach_get('co', verts)
        verts.shape = (vert_count, 3)

        edges = np.empty(edges_count * 2, dtype=np.int32)
        me.edges.foreach_get('vertices', edges)
        edges.shape = (edges_count, 2)

        edge_a = verts[edges[:, 0]]
        edge_b = verts[edges[:, 1]]
        lengths = np.linalg.norm(edge_b - edge_a, axis=1)

        weights = np.zeros(vert_count, dtype=np.float64)
        for v in obj.data.vertices:
            try:
                weights[v.index] = v.groups[0].weight
            except RuntimeError:
                weights[v.index] = 0
            except IndexError:
                weights[v.index] = 0

        print(f'Weights: {len(weights)}')
        print(f'Lengths: {len(lengths)}')
        print(f'Edges: {len(edges)}')
        print(f'Verts: {len(verts)}')

        return verts, edges, 1 - weights, lengths

    @staticmethod
    def read_vertex_pos(obj: bpy.types.Object):
        me: Mesh = obj.data
        vert_count = len(me.vertices)

        verts = np.empty(vert_count * 3, dtype=np.float64)
        me.vertices.foreach_get('co', verts)
        verts.shape = (vert_count, 3)

        return verts

    @staticmethod
    def write_vertex_pos(obj: bpy.types.Object, verts: np.ndarray):
        me: Mesh = obj.data
        count = len(me.vertices)
        verts.shape = count * 3
        me.vertices.foreach_set('co', verts)
        verts.shape = (count, 3)
        me.update()


def get_vertex_degrees(edges):
    return np.unique(edges.flatten(), return_counts=True)[1]


def graph_coloring(edges):
    vertex_count = edges.flatten().max() + 1
    vertex_set = set(range(vertex_count))

    vertex_neighbors = [set() for _ in range(vertex_count)]
    for edge in edges:
        vertex_neighbors[edge[0]].add(edge[1])
        vertex_neighbors[edge[1]].add(edge[0])

    vertex_degrees = get_vertex_degrees(edges)
    min_degree = np.min(vertex_degrees)

    vertex_palette = [{
        'palette': list(range(degree // min_degree)),
        'size': degree // min_degree,
    } for degree in vertex_degrees]

    while vertex_set:
        for vertex in vertex_set:
            random_color = random.choice(vertex_palette[vertex]['palette'])
            vertex_palette[vertex]['color'] = random_color

        # Check if the coloring is valid
        I = set()
        for vertex in vertex_set:
            col = vertex_palette[vertex]['color']
            neighbours_colors = {vertex_palette[neighbour]['color']
                                 for neighbour in vertex_neighbors[vertex]}

            if col not in neighbours_colors:
                I.add(vertex)
                # remove color from all neighbours
                for neighbour in vertex_neighbors[vertex]:
                    if col in vertex_palette[neighbour]['palette']:
                        vertex_palette[neighbour]['palette'].remove(col)

        vertex_set -= I

        for vertex in vertex_set:
            if not vertex_palette[vertex]['palette'] or not I:
                vertex_palette[vertex]['size'] += 1
                vertex_palette[vertex]['palette'] = list(range(vertex_palette[vertex]['size']))

    colors = np.array([vertex['color'] for vertex in vertex_palette])
    print(f'Colors: {len(set(colors))}')
    print(f'Color counts: {np.bincount(colors)}')

    return colors


def constraints_graph(constraint):
    vert_count = np.max(constraint) + 1
    verts_to_edges = {vert_idx: [] for vert_idx in range(vert_count)}

    for edge_idx, edge in enumerate(constraint):
        for vert in edge:
            verts_to_edges[vert].append(edge_idx)

    graph = []
    for edge_idx, edge in enumerate(constraint):
        for vert in edge:
            for other_edge_idx in verts_to_edges[vert]:
                if edge_idx == other_edge_idx:
                    continue
                other_edge = constraint[other_edge_idx]
                if (edge[0] in other_edge) or (edge[1] in other_edge):
                    graph.append(sorted([edge_idx, other_edge_idx]))

    # remove duplicates and convert to numpy array
    return np.array(list(set(tuple(x) for x in graph)))


def is_valid_coloring(edges, coloring):
    return np.all(np.diff(coloring) != 0)
