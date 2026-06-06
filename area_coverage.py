#!/usr/bin/env python3
"""
UAV Coverage Path Planner
Boustrophedon decomposition algorithm for UAV coverage path planning
Compatible with NumPy 2.0+ and Shapely 2.0+
"""

from shapely.geometry import (
    Point, Polygon, LineString, MultiLineString, 
    MultiPoint, GeometryCollection
)
from shapely.errors import TopologicalError, ShapelyError
from shapely.validation import make_valid
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ============================ 常量定义 ============================
DEFAULT_PATH_SPACING = 1.0
ROTATION_OFFSET_DEG = 90
MIN_POLYGON_AREA = 1e-6
LOGGING_LEVEL = logging.INFO

# ============================ 日志配置 ============================
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================ 数据类定义 ============================
@dataclass(frozen=True)
class RotationParams:
    angle_deg: float
    angle_rad: float
    rotation_matrix: np.ndarray  # 纯ndarray，无np.matrix

# =============================================================================
# Rotation Transform Class (完全移除np.matrix)
# =============================================================================
class RotationTransform:
    def __init__(self, angle: float):
        if not isinstance(angle, (int, float)):
            raise TypeError(f"Rotation angle must be numeric, got {type(angle)}")
        self._params = self._compute_rotation_params(angle)

    def _compute_rotation_params(self, angle: float) -> RotationParams:
        angle_rad = np.radians(ROTATION_OFFSET_DEG - angle)
        # ✅ 关键修复1：使用np.array替代np.matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0.0],
            [0.0,                0.0,              1.0]
        ], dtype=np.float64)
        return RotationParams(
            angle_deg=angle,
            angle_rad=angle_rad,
            rotation_matrix=rotation_matrix
        )
    
    @property
    def angle(self) -> float:
        return self._params.angle_deg
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        return self._params.rotation_matrix
    
    def __repr__(self) -> str:
        return f"RotationTransform(angle={self.angle:.1f}°)"

# =============================================================================
# Main Area Polygon Class
# =============================================================================
class AreaPolygon:
    def __init__(
        self,
        coordinates: List[Tuple[float, float]],
        initial_pos: Tuple[float, float],
        interior: Optional[List[List[Tuple[float, float]]]] = None,
        path_spacing: float = DEFAULT_PATH_SPACING,
        fixed_angle: Optional[float] = None
    ):
        interior = interior or []
        self._validate_coordinates(coordinates, "Exterior")
        for idx, hole in enumerate(interior):
            self._validate_coordinates(hole, f"Hole {idx}")
        self._validate_initial_pos(initial_pos)
        
        self.P = self._create_valid_polygon(coordinates, interior)
        if self.P.area < MIN_POLYGON_AREA:
            raise ValueError(f"Polygon area too small: {self.P.area:.6f}")
        
        self.rtf = self._get_rotation_transform(fixed_angle)
        logger.info(f"Using rotation angle: {self.rtf.angle:.1f}°")
        
        self.rP = self._rotate_polygon_safely()
        
        self.origin = self._get_closest_vertex(
            list(self.P.exterior.coords), initial_pos
        )
        logger.info(f"Optimal starting point: {self.origin}")
        
        self.path_spacing = self._validate_path_spacing(path_spacing)

    # -------------------------------------------------------------------------
    # 输入校验方法
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_coordinates(coords: List[Tuple[float, float]], name: str):
        if not isinstance(coords, list) or len(coords) < 3:
            raise ShapelyError(f"{name} must have at least 3 points")
        for (x, y) in coords:
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise TypeError(f"{name} coordinates must be (float, float)")

    @staticmethod
    def _validate_initial_pos(pos: Tuple[float, float]):
        if not isinstance(pos, tuple) or len(pos) != 2:
            raise TypeError("Initial position must be (x, y) tuple")
        if not all(isinstance(p, (int, float)) for p in pos):
            raise TypeError("Initial position coordinates must be numeric")

    @staticmethod
    def _validate_path_spacing(spacing: float) -> float:
        if not isinstance(spacing, (int, float)) or spacing <= 0:
            raise ValueError(f"Path spacing must be positive, got {spacing}")
        return float(spacing)

    @staticmethod
    def _create_valid_polygon(exterior: List[Tuple[float, float]], interior: List[List[Tuple[float, float]]]) -> Polygon:
        try:
            poly = Polygon(exterior, interior)
            return make_valid(poly) if not poly.is_valid else poly
        except (TopologicalError, ShapelyError) as e:
            raise ShapelyError(f"Failed to create valid polygon: {str(e)}")

    # -------------------------------------------------------------------------
    # 旋转变换方法 (完全兼容NumPy 2.0)
    # -------------------------------------------------------------------------
    def _get_rotation_transform(self, fixed_angle: Optional[float]) -> RotationTransform:
        if fixed_angle is not None:
            return RotationTransform(fixed_angle)
        return self._compute_longest_edge_transform()

    def _compute_longest_edge_transform(self) -> RotationTransform:
        coords = list(self.P.exterior.coords)
        n = len(coords)
        
        edge_lengths = []
        for i in range(n):
            p1 = coords[i]
            p2 = coords[(i + 1) % n]
            length = Point(p1).distance(Point(p2))
            edge_lengths.append((length, i, p1, p2))
        
        longest_edge = max(edge_lengths, key=lambda x: x[0])
        logger.info(f"Longest edge at index {longest_edge[1]} (length: {longest_edge[0]:.2f})")
        
        dy = float(longest_edge[3][1] - longest_edge[2][1])
        dx = float(longest_edge[3][0] - longest_edge[2][0])
        
        if dx == 0:
            angle = 90.0 if dy > 0 else -90.0
        else:
            angle = np.degrees(np.arctan(dy / dx))
        
        return RotationTransform(angle)

    def _rotate_points(self, points: np.ndarray) -> np.ndarray:
        """✅ 关键修复2：纯ndarray向量化旋转，无嵌套矩阵"""
        points = np.atleast_2d(points).astype(np.float64)
        # 转换为齐次坐标 (N, 3)
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
        # 矩阵乘法结果为纯ndarray
        rotated_homogeneous = homogeneous @ self.rtf.rotation_matrix.T
        # 提取2D坐标，确保返回(N, 2)形状
        return rotated_homogeneous[:, :2]

    def _rotate_from(self, points: np.ndarray) -> np.ndarray:
        """✅ 关键修复3：使用np.linalg.inv替代matrix.I属性"""
        if not isinstance(points, np.ndarray):
            raise TypeError("rotate_from requires numpy.ndarray input")
        
        points = np.atleast_2d(points).astype(np.float64)
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
        # 计算逆矩阵（ndarray标准方法）
        inv_rot_matrix = np.linalg.inv(self.rtf.rotation_matrix)
        rotated_homogeneous = homogeneous @ inv_rot_matrix.T
        return rotated_homogeneous[:, :2]

    def _rotate_polygon_safely(self) -> Polygon:
        try:
            # 旋转外边界
            exterior_np = np.array(self.P.exterior.coords)
            exterior_rot = self._rotate_points(exterior_np)
            # ✅ 关键修复4：确保坐标是纯Python浮点数元组
            exterior_list = [(float(p[0]), float(p[1])) for p in exterior_rot]
            
            # 旋转内部孔洞
            interiors_list = []
            for hole in self.P.interiors:
                hole_np = np.array(hole.coords)
                hole_rot = self._rotate_points(hole_np)
                hole_list = [(float(p[0]), float(p[1])) for p in hole_rot]
                interiors_list.append(hole_list)
            
            return self._create_valid_polygon(exterior_list, interiors_list)
        except Exception as e:
            raise TopologicalError(f"Failed to rotate polygon: {str(e)}")

    # -------------------------------------------------------------------------
    # 扫掠线生成方法
    # -------------------------------------------------------------------------
    def _generate_base_sweep_line(self) -> LineString:
        min_x, min_y, max_x, max_y = self.rP.bounds
        return LineString([(min_x, min_y), (min_x, max_y)])

    def _generate_single_sweep_line(self, base_line: LineString, offset: float) -> Optional[LineString]:
        try:
            offset_line = base_line.parallel_offset(offset, 'right')
            if not self.rP.intersects(offset_line):
                return None
            
            intersection = self.rP.intersection(offset_line)
            if isinstance(intersection, (GeometryCollection, Point, MultiPoint)):
                return None
            return intersection
        except TopologicalError:
            logger.warning(f"Failed to compute intersection for offset {offset}")
            return None

    def _generate_sweep_lines(self) -> List[LineString]:
        min_x, _, max_x, _ = self.rP.bounds
        base_line = self._generate_base_sweep_line()
        
        try:
            initial_line = self.rP.intersection(base_line)
            lines = [initial_line] if initial_line else []
        except TopologicalError:
            logger.error("Failed to compute initial sweep line")
            return []
        
        num_lines = int((max_x - min_x) / self.path_spacing) + 2
        
        for i in range(1, num_lines):
            offset = i * self.path_spacing
            line = self._generate_single_sweep_line(base_line, offset)
            if not line:
                continue
            
            if isinstance(line, MultiLineString):
                lines.extend(line.geoms)
            else:
                lines.append(line)
        
        valid_lines = [ln for ln in lines if isinstance(ln, LineString) and ln.length > 0]
        logger.info(f"Generated {len(valid_lines)} valid sweep lines")
        return valid_lines

    # -------------------------------------------------------------------------
    # 路径排序方法
    # -------------------------------------------------------------------------
    def _get_closest_vertex(
        self,
        vertices: List[Tuple[float, float]],
        reference: Tuple[float, float]
    ) -> Tuple[float, float]:
        ref_point = Point(reference)
        return min(vertices, key=lambda v: ref_point.distance(Point(v)))

    def _sort_lines_by_distance(self, lines: List[LineString], reference: Tuple[float, float]):
        ref_point = Point(reference)
        lines.sort(key=lambda ln: ref_point.distance(ln))

    def _check_path_obstacle(self, line: LineString) -> bool:
        for hole in self.rP.interiors:
            intersection = hole.intersection(line)
            if isinstance(intersection, LineString) and intersection.length > 0:
                logger.warning("Path intersects with hole")
                return True
        return False

    def _order_sweep_lines(
        self,
        lines: List[LineString],
        start_point: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        waypoints = []
        current_pos = start_point
        remaining_lines = lines.copy()
        
        while remaining_lines:
            self._sort_lines_by_distance(remaining_lines, current_pos)
            current_line = remaining_lines.pop(0)
            
            xs, ys = current_line.xy
            endpoints = list(zip(xs, ys))
            if len(endpoints) < 2:
                logger.warning("Skipping invalid line")
                continue
            
            closest_end = self._get_closest_vertex(endpoints, current_pos)
            farthest_end = endpoints[1] if closest_end == endpoints[0] else endpoints[0]
            
            path_segment = LineString([current_pos, closest_end])
            if self._check_path_obstacle(path_segment):
                continue
            
            waypoints.append(current_pos)
            waypoints.append(closest_end)
            current_pos = farthest_end
        
        # 去重连续重复航点
        if not waypoints:
            return []
        deduped = [waypoints[0]]
        for wp in waypoints[1:]:
            if abs(wp[0] - deduped[-1][0]) > 1e-6 or abs(wp[1] - deduped[-1][1]) > 1e-6:
                deduped.append(wp)
        
        logger.info(f"Generated {len(deduped)} waypoints")
        return deduped

    # -------------------------------------------------------------------------
    # 公共方法
    # -------------------------------------------------------------------------
    def generate_coverage_path(self, custom_origin: Optional[Tuple[float, float]] = None) -> LineString:
        if custom_origin:
            rotated_origin = self._rotate_points(np.array([custom_origin]))[0]
            rotated_origin = (float(rotated_origin[0]), float(rotated_origin[1]))
        else:
            rotated_origin = self._rotate_points(np.array([self.origin]))[0]
            rotated_origin = (float(rotated_origin[0]), float(rotated_origin[1]))
        
        sweep_lines = self._generate_sweep_lines()
        if not sweep_lines:
            raise RuntimeError("No valid sweep lines generated")
        
        waypoints_rotated = self._order_sweep_lines(sweep_lines, rotated_origin)
        if not waypoints_rotated:
            raise RuntimeError("No valid waypoints generated")
        
        waypoint_array = np.array(waypoints_rotated, dtype=np.float64)
        waypoints_original = self._rotate_from(waypoint_array)
        
        # 确保最终坐标是纯浮点数
        final_waypoints = [(float(p[0]), float(p[1])) for p in waypoints_original]
        final_path = LineString(final_waypoints)
        
        if not final_path.is_valid:
            final_path = make_valid(final_path)
        
        logger.info(f"Final path length: {final_path.length:.2f} meters")
        return final_path

# =============================================================================
# 可视化工具
# =============================================================================
def plot_coordinates(ax, geometry, marker_size: int = 3):
    x, y = geometry.xy
    ax.plot(x, y, 'o', color='#999999', markersize=marker_size, zorder=1)

def plot_path(ax, geometry, color: str = 'blue', linewidth: float = 2.0, alpha: float = 0.7):
    x, y = geometry.xy
    ax.plot(
        x, y,
        alpha=alpha,
        linewidth=linewidth,
        solid_capstyle='round',
        color=color,
        zorder=2
    )

# =============================================================================
# 主执行逻辑
# =============================================================================
if __name__ == '__main__':
    logger.info("=== UAV Coverage Path Planner ===")
    
    # 测试多边形（带有效孔洞）
    exterior = [(0, 0), (4, 4), (0, 8), (-4, 4), (-9, 3)]
    holes = [[(-2, 3), (-1, 4), (0, 3), (-1, 2)]]  # 有效四边形孔洞
    
    try:
        polygon = AreaPolygon(
            coordinates=exterior,
            initial_pos=(-5, 10),
            interior=holes,
            path_spacing=0.5,
            fixed_angle=30
        )
        
        coverage_path = polygon.generate_coverage_path(custom_origin=(0.0, 0.0))
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=90)
        
        # 旋转坐标系
        ax1.plot(*polygon.rP.exterior.xy, 'b-', linewidth=2, label='Rotated Area')
        for hole in polygon.rP.interiors:
            ax1.plot(*hole.xy, 'k-', linewidth=1)
        plot_path(ax1, coverage_path, color='red')
        plot_coordinates(ax1, coverage_path)
        ax1.set_title("Rotated Coordinate System")
        ax1.set_aspect('equal', adjustable='box')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 原始坐标系
        ax2.plot(*polygon.P.exterior.xy, 'g--', linewidth=2, label='Original Area')
        for hole in polygon.P.interiors:
            ax2.plot(*hole.xy, 'k-', linewidth=1)
        plot_path(ax2, coverage_path, color='red')
        plot_coordinates(ax2, coverage_path)
        ax2.set_title("Original Coordinate System")
        ax2.set_aspect('equal', adjustable='box')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        fig.suptitle("UAV Coverage Path Planning (Boustrophedon Decomposition)", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        logger.info("Path generation completed successfully!")
        logger.info(f"Total waypoints: {len(list(coverage_path.coords))}")
        
    except Exception as e:
        logger.error(f"Failed to generate path: {str(e)}", exc_info=True)