import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
from threading import Thread
import time


class Open3DVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("3D Human Pose", 800, 600)
        self.points = None
        self.lines = o3d.geometry.LineSet()
        self.pcd = o3d.geometry.PointCloud()
        self.conn = mp.solutions.pose.POSE_CONNECTIONS

    def update(self, landmarks):
        if not landmarks: return
        pts = [[p.x, -p.y, -p.z] for p in landmarks.landmark]
        self.points = np.array(pts)

        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * 17)

        line_idx = [[s, e] for s, e in self.conn]
        self.lines.points = o3d.utility.Vector3dVector(self.points)
        self.lines.lines = o3d.utility.Vector2iVector(line_idx)
        self.lines.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(line_idx))

    def run(self):
        while True:
            self.vis.clear_geometries()
            if self.points is not None:
                self.vis.add_geometry(self.pcd)
                self.vis.add_geometry(self.lines)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)


def main():
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    vis = Open3DVisualizer()
    Thread(target=vis.run, daemon=True).start()

    with mp_pose.Pose(model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                vis.update(res.pose_landmarks)

            cv2.putText(frame, "3D Human Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("2D Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()