#pragma once

#include <pangolin/pangolin.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>   
#else
#include <GL/gl.h>
#endif
#include <Eigen/Dense>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

namespace vio {

class VioVisualizer {
public:
    VioVisualizer() : running_(true) {}

    void addEstimate(const Eigen::Vector3d& p) {
        std::lock_guard<std::mutex> lock(mtx_);
        est_path_.push_back(p);
    }

    void addGroundTruth(const Eigen::Vector3d& p) {
        std::lock_guard<std::mutex> lock(mtx_);
        gt_path_.push_back(p);
    }

    void stop() {
        running_ = false; 
    }

    void run() {
        pangolin::CreateWindowAndBind("VIO Trajectory Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -10, -15, 0, 0, 0, 0.0, -1.0, 0.0)
        );

        pangolin::Handler3D handler(s_cam);
        pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(&handler);

        while (!pangolin::ShouldQuit() && running_) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

            d_cam.Activate(s_cam);

            std::unique_lock<std::mutex> lock(mtx_);
            std::vector<Eigen::Vector3d> local_gt = gt_path_;
            std::vector<Eigen::Vector3d> local_est = est_path_;
            lock.unlock();

            glLineWidth(2.0);
            glColor3f(1.0f, 0.0f, 0.0f); // GT red
            drawPath(local_gt);

            glLineWidth(2.0);
            glColor3f(0.0f, 1.0f, 0.0f); // Estimate green
            drawPath(local_est);

            pangolin::FinishFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }
    bool isRunning() const { return running_; }

private:
    void drawPath(const std::vector<Eigen::Vector3d>& path) {
        if (path.empty()) return;
        glBegin(GL_LINE_STRIP);
        for (const auto& p : path) {
            glVertex3f(p.x(), p.y(), p.z());
        }
        glEnd();
    }

    std::vector<Eigen::Vector3d> est_path_;
    std::vector<Eigen::Vector3d> gt_path_;
    std::mutex mtx_;
    std::atomic<bool> running_;
};

} // namespace vio