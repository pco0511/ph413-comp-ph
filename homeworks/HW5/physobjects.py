from manim import *

from typing import Any

from typing_extensions import Literal, Self, TypeAlias

from manim.typing import Point2DLike, Point3D, Point3DLike, Vector3D
from manim.utils.color import ParsableManimColor

SpringEdgeType = Literal['edged', 'round']
    
    
class Spring(Polygram):
    def __init__(
        self,
        start: Point3DLike | Mobject = LEFT,
        end: Point3DLike | Mobject = RIGHT,
        spring_width: float = 0.5,
        num_turns: int = 4,
        uncoiled: float = 0.2,
        spring_shape: SpringEdgeType = 'edged',
        **kwargs: Any,
    ) -> None:
        self.start=np.array(start),
        self.end=np.array(end),
        self.dim = 3
        self.spring_width = spring_width
        self.num_turns = num_turns
        self.uncoiled = uncoiled
        self.spring_shape = spring_shape
        
        points = self.generate_vertices(
            start=self.start,
            end=self.end,
            spring_width=self.spring_width,
            num_turns=self.num_turns,
            uncoiled=self.uncoiled
        )
        
        super().__init__(
            points,
            **kwargs
        )
        
        
        
    def generate_vertices(
        self,
        start: Point3DLike | Mobject,
        end: Point3DLike | Mobject,
        spring_width: float = 0.5,
        num_turns: int = 4,
        uncoiled: float = 0.2,
        spring_shape: SpringEdgeType = 'edged',
    ) -> None:
        """Sets the points of the spring based on its start and end points.

        Unlike :meth:`put_start_and_end_on`, this method respects `self.buff` and
        Mobject bounding boxes.

        Parameters
        ----------
        start : Point3DLike or Mobject
            The start point or Mobject of the spring.
        end : Point3DLike or Mobject
            The end point or Mobject of the spring.
        width : float, optional
            Thickness of the spring coil, by default 0.5.
        num_turns : int, optional
            Number of helical turns in the spring, by default 4.
        uncoiled : float, optional
            Ratio (0 to 1) of the spring that remains uncoiled (i.e., straight), by default 0.2.
        spring_shape : SpringEdgeType, optional
            Shape of the spring (e.g., 'edged' or 'smooth'), by default 'edged'.
        buff : float, optional
            Empty space to leave at both ends between the spring and the objects, by default 0.
        path_arc : float, optional
            Angle (in radians) to arc the path of the spring, by default 0 (a straight line).
        """
        assert 0 <= uncoiled <= 1
        
        vector = np.array(end) - np.array(start)
        x, y, _ = vector[0] / np.linalg.norm(vector)
        normal = (spring_width / 2) * np.array([-y, x, 0])
        points = []
        
        points.append(np.array(start))
        
        cursor = np.array(start)
        cursor += (uncoiled / 2) * vector

        points.append(cursor.copy())
        
        coiled = 1 - uncoiled
        quater_unit = (coiled / num_turns) / 4
        
        for _ in range(num_turns):
            cursor += quater_unit * vector
            points.append(cursor + normal)
            cursor += quater_unit * vector
            points.append(cursor.copy())
            cursor += quater_unit * vector
            points.append(cursor - normal)
            cursor += quater_unit * vector
            points.append(cursor.copy())
            
        points.append(np.array(end))
        
        return points


class SpringTestScene(Scene):
    def construct(self):
        spring = Spring(2 * LEFT, 2 * RIGHT, )
        self.add(spring)