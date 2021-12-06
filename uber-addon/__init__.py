# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import bpy
from .uber_soft import *

classes = (
    SOFT_OT_Action,
    SOFT_PT_Panel
)

bl_info = {
    "name": "uber_soft",
    "author": "DrTtnk",
    "description": "",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "",
    "category": "Generic"
}

soft_instance = None


def render_pre(scene, b):

    global soft_instance
    obj: Object = bpy.context.active_object

    if soft_instance is None and obj is not None and obj.type == 'MESH':
        soft_instance = Soft(obj)

    with funcy.log_durations(print):
        soft_instance.update()

    print("render_pre")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
