
import time
import threading


class dummyGui:
    ROOT = 0
    COLAPSIING_HEADER = 1
    TABLE = 2
    TEXT = 3
    IMAGE = 4
    def visualize(self, path:str, type:int, data:object=None, opened:bool=False):
        pass

t:threading.Thread = None
GUI = dummyGui()
WM = None

def import_all():

# |====================================================================================================================
# | START OF THE REAL SCRIPT
# |====================================================================================================================


    global GUI, WM, t

    from _Utils.Color import prntC
    import _Utils.Color as C
    import ctypes as T
    from Xlib import display
    import Xlib as X
    import PIL

    Xlib = T.CDLL("libX11.so.6")
    Window = T.c_ulong
    root_disp = Xlib.XOpenDisplay(None)
    root = Xlib.XDefaultRootWindow(root_disp)
    _OLM = 2**32-1

    class WindowManager():

        def __init__(self) -> None:
            self.display = display.Display()
            self.root = self.display.screen().root

        def getWindowsList(self):
            ids = self.root.get_full_property(self.display.get_atom('_NET_CLIENT_LIST'), 0).value
            windows = {}
            for id in ids:
                window = self.display.create_resource_object('window', id)
                # check window is valid
                try:
                    name = window.get_full_property(self.display.get_atom('_NET_WM_NAME'), 0)
                except:
                    print("window in invalid state")
                    continue
                name = name.value
                windows[id] = name

            return windows




        def closeWindow(self, id):
            Xlib.XDestroyWindow(root_disp, id)
            Xlib.XCloseDisplay(root_disp)




    ##############################################################################""



    import dearpygui.dearpygui as dpg
    import folium

    import os



    # |====================================================================================================================
    # | TREE STRUCTURE
    # |====================================================================================================================
    # if (False):
    #     from .DebugGui import Node
    class Node: pass
    class Node:
        fast_access = {}

    # |--------------------------------------------------------------------------------------------------------------------
    # |     INITIALIZATION
    # |--------------------------------------------------------------------------------------------------------------------

        def __init__(self, parent:Node, tag:str, type:int, data=None) -> None:
            self.parent = parent
            self.tag = tag
            self.type = type
            self.data = data
            self.path = self.__compute_my_path__()
            self.childrens:dict[str, Node] = {}

        def __compute_my_path__(self) -> str:
            if (self.parent == None):
                self.path = self.tag
            else:
                self.path = self.parent.path + "/" + self.tag
            return self.path

    # |--------------------------------------------------------------------------------------------------------------------
    # |     ADD CHILD
    # |--------------------------------------------------------------------------------------------------------------------

        def add_child(self, tag:str, type:int, data=None) -> Node:
            child = Node(self, tag, type, data)

            self.childrens[child.tag] = child
            Node.fast_access[child.path] = child

            return child

        def pop(self, tag:str)->None:
            if (tag in self.childrens):
                del Node.fast_access[self.childrens[tag].path]
                del self.childrens[tag]



    # |--------------------------------------------------------------------------------------------------------------------
    # |    GETTERS
    # |--------------------------------------------------------------------------------------------------------------------

        def __get__(self, path:str) -> Node:
            if ("/" in path):
                path = path.strip("/ \t")
                return Node.fast_access.get(path)
            else:
                return self.childrens.get(path)

        def get(self, path:str, default=None) -> Node:
            res = self.__get__(path)
            if (res == None):
                return default
            return res

        def __getitem__(self, key:str) -> Node:
            return self.__get__(key)


        def __contains__(self, key:str) -> bool:
            return self.__get__(key) != None

        def __str__(self) -> str:
            return self.path


    # |====================================================================================================================
    # | DEBUG GUI CLASS
    # |====================================================================================================================

    # Decorator to check if DEBUG_GUI is still open
    def open(func):
        def wrapper(self, *args, **kwargs):
            if (self.uuid == None):
                if (self.first_error):
                    prntC(C.WARNING, "Debug GUI has been closed")
                    self.first_error = False
                return
            return func(self, *args, **kwargs)
        return wrapper


    class DebugGui:

        __TYPE_LABEL__ = ["ROOT", "COLAPSIING_HEADER", "TABLE", "TEXT", "IMAGE", "__TABLE_ROW__", "__TABLE_COLUMN__"]
        ROOT = 0
        COLAPSIING_HEADER = 1
        TABLE = 2
        TEXT = 3
        IMAGE = 4
        __TABLE_ROW__ = 5
        __TABLE_COLUMN__ = 6

        SHOW_DEBUG = True


        def __init__(self)->None:
            self.uuid = None
            self.tree = None
            self.first_error = True
            self.tree = Node(None, "MainWindow", DebugGui.ROOT)
            self.textures = set()

        def init(self) -> None:
            dpg.create_context()
            dpg.create_viewport(title="Debug GUI", width=1000, height=800)

            with dpg.window(width=1000, height=800,
                            tag="MainWindow", label="MainWindow",
                            no_resize=True, no_move=True, no_title_bar=True):

                dpg.set_item_pos("MainWindow", (0, 0))


            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.start_dearpygui()


        def __show_debug__(self) -> None:
            print("show debug")
            self.visualize("__DEBUG__/MainWindow", DebugGui.COLAPSIING_HEADER)


        def run(self) -> None:
            prntC(C.INFO, "Starting Debug GUI")
            self.init()
            dpg.destroy_context()
            WM.closeWindow(self.uuid)
            self.uuid = None
            prntC(C.INFO, "Debug GUI closed")



        @open
        def visualize(self, path:str, type:int, data:object=None, opened:bool=False):
            # remove first and last slash
            path = path.strip("/ \t").split("/")

    # |--------------------------------------------------------------------------------------------------------------------
    # |     Step 1 : walk unti the last node of the path. If some nodes are missing, create them with a
    # |              logical element, or a collapsing header by default
    # |--------------------------------------------------------------------------------------------------------------------

            node_actu = self.tree

            for c in range(len(path)-1):
                child_name = path[c]
                child = node_actu.get(child_name)
                if (child == None):
                    self.__create_child_auto__(node_actu, child_name, open_colapsing_header=opened)

                node_actu = node_actu[child_name]


            if (type == DebugGui.COLAPSIING_HEADER):
                data = opened

            self.__update_child__(node_actu, path[-1], type, data)


    # |--------------------------------------------------------------------------------------------------------------------
    # |     Child creation (auto or not)
    # |--------------------------------------------------------------------------------------------------------------------


        def __create_child_auto__(self, parent_node:Node, child_name:str, open_colapsing_header:bool=False):

            if (parent_node.type in [DebugGui.ROOT, DebugGui.COLAPSIING_HEADER]):
                self.__create_collapsing_header__(parent_node, child_name, open_colapsing_header)
            elif (parent_node.type == DebugGui.TABLE):
                self.__create_table_column__(parent_node, child_name)
            elif (parent_node.type == DebugGui.__TABLE_COLUMN__):
                self.__create_table_row__(parent_node, child_name)
            else:
                prntC(C.ERROR, "Cannot add child to a",C.BLUE, DebugGui.__TYPE_LABEL__[parent_node.type],C.RESET,"node")

        def __create_child__(self, parent_node:Node, child_name:str, type:int, data:object=None):
            if (type == DebugGui.COLAPSIING_HEADER):
                self.__create_collapsing_header__(parent_node, child_name, data)
            elif (type == DebugGui.TABLE):
                self.__create_table__(parent_node, child_name, data)
            elif (type == DebugGui.IMAGE):
                self.__create_image__(parent_node, child_name, data)
            elif (type == DebugGui.TEXT):
                self.__create_text__(parent_node, child_name, data)
            else:
                prntC(C.WARNING, "Unknown widget type", C.BLUE, type)



    # |--------------------------------------------------------------------------------------------------------------------
    # |     Each type of node has its own create function
    # |--------------------------------------------------------------------------------------------------------------------

        def __create_collapsing_header__(self, parent_node:Node, child_name:str, opened:bool=False):
            # print("Create collapsing header", c hild_name, "under", parent_node)
            child = parent_node.add_child(child_name, DebugGui.COLAPSIING_HEADER)
            indent = child.path.count("/")-1
            # indent = "\t"*indent
            dpg.add_collapsing_header(label=child_name, tag=child.path, parent=parent_node.path,
                                    indent=indent, default_open=opened)

            self.__update_debug__(child)

        def __create_table__(self, parent_node:Node, child_name:str, columns:int):
            child = parent_node.add_child(child_name, DebugGui.TABLE, {"rows":0, "columns":columns})
            print("Create table", child_name, "under", parent_node, "tag:", child.path)
            dpg.add_table(header_row=False, tag=child.path, parent=parent_node.path)

            for c in range(columns):
                print("Create column", child.path+f"/{c}", "under", child.path)
                dpg.add_table_column(tag=child.path+f"/{c}", parent=child.path)

            self.__update_debug__(child)


        def __create_table_column__(self, parent_node:Node, child_name:str):
            # print("Create table_column", child_name, "under", parent_node)
            child = parent_node.add_child(child_name, DebugGui.__TABLE_COLUMN__)

            actual_columns = parent_node.data["columns"]
            parent_node.data["columns"] = max(parent_node.data["columns"], int(child_name)+1)

            print("Actual columns", actual_columns, "new columns", parent_node.data["columns"])

            for c in range(actual_columns, parent_node.data["columns"]):
                print("Create column", parent_node.path+f"/{c}", "under", parent_node.path)
                dpg.add_table_column(tag=parent_node.path+f"/{c}", parent=parent_node.path)

                for r in range(parent_node.data["rows"]):
                    print("Create cell", parent_node.parent.path+f"/{c}/{r}")
                    dpg.add_table_cell(tag=parent_node.path+f"/{c}/{r}", parent=parent_node.path+f"/#/{r}")

            self.__update_debug__(child)


        def __create_table_row__(self, parent_node:Node, child_name:str):
            # print("Create table_row", child_name, "under", parent_node)
            child = parent_node.add_child(child_name, DebugGui.__TABLE_ROW__)
            # dpg.add_table_row(label=child_name, tag=child.path, parent=parent_node.parent.path)

            actual_rows = parent_node.parent.data["rows"]
            parent_node.parent.data["rows"] = max(parent_node.parent.data["rows"], int(child_name)+1)

            print("Actual rows", actual_rows, "new rows", parent_node.parent.data["rows"])

            for r in range(actual_rows, parent_node.parent.data["rows"]):
                print("Create row", parent_node.parent.path+f"/#/{r}", "under", parent_node.parent.path)
                dpg.add_table_row(tag=parent_node.parent.path+f"/#/{r}", parent=parent_node.parent.path)

                for c in range(parent_node.parent.data["columns"]):
                    print("Create cell", parent_node.parent.path+f"/{c}/{r}", "under", parent_node.parent.path+f"/#/{r}")
                    dpg.add_table_cell(tag=parent_node.parent.path+f"/{c}/{r}", parent=parent_node.parent.path+f"/#/{r}")

            self.__update_debug__(child)


        def get_available_width(self, node:Node):
            width = 1000
            while (node.parent != None):
                if (node.type == DebugGui.TABLE):
                    width /= node.data["columns"]

                node = node.parent
            return min(width, 500)


        def __create_image__(self, parent_node:Node, child_name:str, path:str):
            print("Create image", child_name, "under", parent_node)
            child = parent_node.add_child(child_name, DebugGui.IMAGE, path)

            width, height, channels, data = dpg.load_image(path)
            with dpg.texture_registry(show=False):
                dpg.add_static_texture(width=width, height=height, default_value=data, tag=path)
                self.textures.add(path)

            available_width = self.get_available_width(parent_node)
            height = (height/width)*available_width
            dpg.add_image(label=child_name, texture_tag=path,
                            width=available_width, height=height,
                        tag=child.path, parent=parent_node.path)

            self.__update_debug__(child)

        def __create_text__(self, parent_node:Node, child_name:str, text:str):
            print("Create text", child_name, "under", parent_node)
            child = parent_node.add_child(child_name, DebugGui.TEXT, text)
            dpg.add_text(text, tag=child.path, parent=parent_node.path)

            self.__update_debug__(child)


        def __update_debug__(self, node:Node):
            if (DebugGui.SHOW_DEBUG and not(node.path.startswith("MainWindow/__DEBUG__"))):
                self.visualize("__DEBUG__/"+node.path, DebugGui.COLAPSIING_HEADER)

    # |--------------------------------------------------------------------------------------------------------------------
    # |     Updating a node :
    # |--------------------------------------------------------------------------------------------------------------------


        def __update_child__(self, parent_node:Node, child_name:str, type:int, data:object):
            if (parent_node.get(child_name) == None):
                self.__create_child__(parent_node, child_name, type, data)
                return

            if (parent_node[child_name].type != type):
                # remove the child and create a new one
                self.__remove_child__(parent_node, child_name)
                self.__create_child__(parent_node, child_name, type, data)
                return

            if (type == DebugGui.TEXT):
                self.__update_text__(parent_node[child_name], data)
            elif (type == DebugGui.IMAGE):
                self.__update_image__(parent_node[child_name], data)


        def __update_text__(self, node:Node, text:str):
            dpg.set_value(node.path, text)


        def __update_image__(self, node:Node, path:str):
            tex = node.data
            self.textures.remove(tex)
            dpg.delete_item(tex)
            dpg.delete_item(node.path)

            self.__create_image__(node.parent, node.tag, path)


    # |--------------------------------------------------------------------------------------------------------------------
    # |    Removing a node
    # |--------------------------------------------------------------------------------------------------------------------


        def __remove_child__(self, parent_node:Node, child_name:str):
            dpg.delete_item(parent_node[child_name].path)
            parent_node.pop(child_name)




    GUI = DebugGui()
    WM = WindowManager()






active = False
def activate():
    global active
    active = True


def __set_uuid__():
    global GUI
    while (GUI.uuid == None):
        time.sleep(0.1)
        windows = WM.getWindowsList()
        for id in windows:
            if (windows[id] == "Debug GUI"):
                GUI.uuid = id
                break

def __run_gui__():
    global GUI
    GUI.run()

def launch_gui(CTX):
    if not(active): return
    import_all()
    global GUI, t
    # Call debug gui in a separate thread
    t = threading.Thread(target=__run_gui__)
    t.start()

    get_uuid = threading.Thread(target=__set_uuid__)
    get_uuid.start()
    get_uuid.join()

    GUI.__show_debug__()






