"""Tool integration tests with transport/model dependencies isolated from ROS."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from .localization import LocalizationManager


def _load_tools():
    # This repository's lightweight test environment has no ROS/LangChain stack.
    # Exercise the real tool methods; stub only external transport/model wrappers.
    class Model:
        def __init__(self, **values):
            self.__dict__.update(values)

        def model_copy(self, *, update):
            result = copy.copy(self)
            result.__dict__.update(update)
            return result

        def model_dump(self):
            return dict(self.__dict__)

    class Tool(Model):
        def invoke(self, values):
            return self._run(**values)

    modules = {name: ModuleType(name) for name in (
        'pydantic', 'langchain_core', 'langchain_core.tools',
        'rai', 'rai.communication', 'rai.communication.ros2',
        'rai.communication.ros2.connectors',
    )}
    modules['pydantic'].BaseModel = Model
    modules['pydantic'].Field = lambda default=None, **kwargs: default
    modules['pydantic'].model_validator = lambda **kwargs: lambda method: method
    modules['langchain_core.tools'].BaseTool = Tool
    modules['rai.communication.ros2'].ROS2Message = Model
    modules['rai.communication.ros2.connectors'].ROS2Connector = object
    name = 'examples.pc_agent._tools_localization_test_impl'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('tools.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def _pose():
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='map'),
        pose=SimpleNamespace(covariance=[0.0] * 36, pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )),
    )


def _transform():
    return SimpleNamespace(transform=SimpleNamespace(
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
    ))


class Connector:
    def __init__(self):
        self.callbacks = {}
        self.goals = []
        self.tf_calls = []
        self.tf_missing = False
        self.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: 'stamp')))

    def register_callback(self, source, callback, **kwargs):
        self.callbacks[source] = callback
        return source

    def get_transform(self, **kwargs):
        self.tf_calls.append(kwargs)
        if self.tf_missing and kwargs['target_frame'] == 'map':
            raise RuntimeError('map frame does not exist')
        return _transform()

    def start_action(self, **kwargs):
        self.goals.append(kwargs)
        return 'navigation-1'

    def send_message(self, *args, **kwargs):
        raise AssertionError('Must not publish rotation or initialpose')

    def service_call(self, *args, **kwargs):
        raise AssertionError('Must not call localization services')


class ToolLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = _load_tools()

    def setUp(self):
        self.tools.clear_detection_history()
        self.connector = Connector()
        self.manager = LocalizationManager(self.connector)
        self.navigation = self.tools.NavigateToCoordinatesTool(
            connector=self.connector, localization_manager=self.manager)
        self.detection = self.tools.GetDetectionsTool(
            connector=self.connector, localization_manager=self.manager)

    def acquire(self):
        self.connector.callbacks['/initialpose'](_pose())
        self.manager.ensure_localized()

    def detection_object(self):
        return self.tools.DetectionObject(
            class_name='chair', x=4.0, y=2.0, z=0.4, size_x=0.4,
            size_y=0.4, confidence=0.91, direction='正前方', confirmed_hits=3)

    def run_socket_detection(self):
        with patch.object(self.tools, '_get_socket_client', return_value=object()), patch.object(
            self.tools.GetDetectionsTool, '_stabilize_socket_payload',
            return_value=({'frame_id': 'rslidar'}, [self.detection_object()]),
        ):
            return self.detection._run()

    def test_manual_pose_can_navigate_without_amcl(self):
        self.acquire()
        result = self.navigation._run(5.0, 2.0)
        self.assertIn('导航已开始', result)
        self.assertEqual(len(self.connector.goals), 1)
        self.assertEqual(self.connector.tf_calls[-1]['target_frame'], 'map')
        self.assertEqual(self.connector.tf_calls[-1]['source_frame'], 'base_link')
        self.assertEqual(self.manager.get_status()['pose']['source'], '/initialpose')

    def test_missing_tf_blocks_goal_without_clearing_localization(self):
        self.acquire()
        self.connector.tf_missing = True
        result = self.navigation._run(5.0, 2.0)
        self.assertIn('已定位，但导航坐标变换不可用', result)
        self.assertIn('map frame does not exist', result)
        self.assertIn('无需重新发布', result)
        self.assertEqual(self.connector.goals, [])
        self.assertTrue(self.manager.is_localized())

    def test_uninitialized_navigation_does_not_query_tf_or_send_goal(self):
        result = self.navigation._run(5.0, 2.0)
        self.assertIn('获取初始坐标', result)
        self.assertEqual(self.connector.goals, [])
        self.assertEqual(self.connector.tf_calls, [])

    def test_tf_failure_keeps_relative_detection_but_no_navigation_snapshot(self):
        self.acquire()
        self.connector.tf_missing = True
        result = self.run_socket_detection()
        self.assertIn('椅子', result)
        self.assertIn('方向汇总', result)
        self.assertIn('置信度=0.91', result)
        self.assertIn('已定位', result)
        self.assertIn('map frame does not exist', result)
        self.assertNotIn('x=', result)
        self.assertNotIn('y=', result)
        self.assertEqual(self.tools.get_detection_snapshot(), [])
        self.assertEqual(self.tools.get_latest_detection_round()['status'], 'failed')
        self.assertTrue(self.manager.is_localized())

    def test_failed_round_never_reuses_previous_map_coordinates(self):
        self.acquire()
        self.run_socket_detection()
        self.assertEqual(len(self.tools.get_detection_snapshot()), 1)
        self.connector.tf_missing = True
        self.run_socket_detection()
        target = self.tools.NavigateToDetectedTargetTool(navigate_tool=self.navigation)
        result = target._run('chair')
        self.assertIn('没有可用的结构化快照', result)
        self.assertEqual(self.connector.goals, [])

    def test_detected_target_uses_same_manual_pose_without_amcl(self):
        self.acquire()
        self.run_socket_detection()
        target = self.tools.NavigateToDetectedTargetTool(navigate_tool=self.navigation)
        with patch.object(self.tools.NavigateToDetectedTargetTool, '_load_global_costmap',
                          side_effect=RuntimeError('costmap unavailable')):
            result = target._run('chair')
        self.assertIn('导航已开始', result)
        self.assertEqual(len(self.connector.goals), 1)
        self.assertTrue(self.manager.is_localized())

    def test_detected_target_also_requires_live_tf(self):
        self.acquire()
        self.run_socket_detection()
        self.connector.tf_missing = True
        target = self.tools.NavigateToDetectedTargetTool(navigate_tool=self.navigation)
        with patch.object(self.tools.NavigateToDetectedTargetTool, '_load_global_costmap',
                          side_effect=RuntimeError('costmap unavailable')):
            result = target._run('chair')
        self.assertIn('导航坐标变换不可用', result)
        self.assertEqual(self.connector.goals, [])
        self.assertTrue(self.manager.is_localized())

    def test_uninitialized_detection_error_reports_initialization_separately(self):
        self.connector.tf_missing = True
        result = self.run_socket_detection()
        self.assertIn('尚未获取初始坐标', result)
        self.assertIn('检测坐标变换', result)
        self.assertNotIn('AMCL 已完成', result)

    def test_ros_path_preserves_relative_perception_without_map_tf(self):
        self.acquire()
        self.connector.tf_missing = True
        self.detection.detection_source = 'ros'
        message = SimpleNamespace(header=SimpleNamespace(frame_id='rslidar'))
        node = SimpleNamespace(destroy_node=lambda: None)
        def subscribe(_type, _topic, callback, _depth):
            callback(message)
        node.create_subscription = subscribe
        ros = ModuleType('rclpy')
        ros.create_node = lambda name: node
        ros.spin_once = lambda *args, **kwargs: None
        vision = ModuleType('vision_msgs.msg')
        vision.Detection3DArray = object
        with patch.dict(sys.modules, {'rclpy': ros, 'vision_msgs': ModuleType('vision_msgs'),
                                      'vision_msgs.msg': vision}), patch.object(
            self.tools.GetDetectionsTool, '_parse_detection3d_array',
            return_value=[self.detection_object()]), patch.object(
            self.tools.DetectionStabilizer, 'update', return_value=[self.detection_object()]):
            result = self.detection._run()
        self.assertIn('椅子', result)
        self.assertIn('暂无可导航', result)
        self.assertNotIn('x=', result)
        self.assertEqual(self.tools.get_detection_snapshot(), [])


if __name__ == '__main__':
    unittest.main()
