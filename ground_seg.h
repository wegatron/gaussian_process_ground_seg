#ifndef GROUND_SEG_H
#define GROUND_SEG_H

#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace np = boost::python::numpy;

struct ground_seg_params {
    // variables
    float rmax = 100;         // max radius of point to consider

    // for GP model
    float p_l = 32;  // length parameter, how close points have to be in the GP model to correlate them
    float p_sf = 0.5;    // scaling on the whole covariance function
    float p_sn = 0.01;  // the expected noise for the mode

    float p_tmodel = 5;  // the required confidence required in order to consider something ground
    float p_tdata = 0.1;  // scaled value that is required for a query point to be
    float p_tg = 0.3;
    // seeding parameters
    float max_seed_range = 30;   // meters
    float max_seed_height = 0.3;  // meters
    float mount_angle = 0;

    int num_bins_a = 180;
    int num_bins_l = 160;
};

struct polar_bin_cell
{
    std::vector<int> cell_pt_inds;

    // proto ground point
    int index;
    float height;
    float r;
    bool is_ground;
};

class gaussian_process_ground_seg
{
public:
    gaussian_process_ground_seg(const std::string &config_file);
    gaussian_process_ground_seg(const ground_seg_params &params)
        : params_(params){}

    /**
     * \brief segment point cloud, return labels, 1 for ground and 0 for nonground
     */
    std::vector<uint8_t> segment(pcl::PointCloud<pcl::PointXYZ>::Ptr &pts);

    /**
     * @brief as segment, a python interface.
     */
    np::ndarray segment_py(const np::ndarray &pts);
private:
    void gen_polar_bin_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr &pts);
    void extract_seed(const int ind,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts,
                      std::vector<polar_bin_cell*> &current_model,
                      std::vector<polar_bin_cell*> &sig_pts);
    Eigen::MatrixXd gp_kernel(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1);
    void insac(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts,
               const int max_iter, std::vector<polar_bin_cell*> &model,
               std::vector<polar_bin_cell*> &sig_pts);
    void label_pc(const int ind, const pcl::PointCloud<pcl::PointXYZ>::Ptr pts, std::vector<uint8_t> &labels);
    ground_seg_params params_;
    std::vector<polar_bin_cell> polar_bins_;
};
#endif //GROUND_SEG_H
