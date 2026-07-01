import numpy as np

try:
    import vtkmodules.vtkInteractionStyle as vtk_style
    BaseStyle = vtk_style.vtkInteractorStyleTrackballCamera
except (ImportError, AttributeError):
    import vtk
    BaseStyle = vtk.vtkInteractorStyleTrackballCamera


def format_value(v):
    if v == 0:
        return "0"
    elif abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.2e}"
    else:
        return f"{v:.4g}"


class CutPlaneStyle(BaseStyle):
    def __init__(self, editor):
        super().__init__()
        self._editor = editor
        self.AddObserver("MouseWheelForwardEvent", self.fwd)
        self.AddObserver("MouseWheelBackwardEvent", self.bwd)

    def fwd(self, obj, event):
        self._editor.on_wheel(1)

    def bwd(self, obj, event):
        self._editor.on_wheel(-1)