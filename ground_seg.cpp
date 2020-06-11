#include "ground_seg.h"
#include <math.h>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
//#include <zsw_vtk_io.h>

gaussian_process_ground_seg::gaussian_process_ground_seg(const std::string &config_file)
{
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    fs["rmax"] >> params_.rmax;
    fs["p_l"] >> params_.p_l;
    fs["p_sf"] >> params_.p_sf;
    fs["p_sn"] >> params_.p_sn;
    fs["p_tmodel"] >> params_.p_tmodel;
    fs["p_tdata"] >> params_.p_tdata;
    fs["p_tg"] >> params_.p_tg;
    fs["max_seed_range"] >> params_.max_seed_range;
    fs["max_seed_height"] >> params_.max_seed_height;
    fs["mount_angle"] >> params_.mount_angle;
    fs["num_bins_a"] >> params_.num_bins_a;
    fs["num_bins_l"] >> params_.num_bins_l;
}

/**
 * @brief 将点云按照水平角度和距离划分成polar bin cells, 对于每一个cell, 抽取其最低点作为代表(将来进行高斯过程fitting)
 */
void gaussian_process_ground_seg::gen_polar_bin_grid(
     pcl::PointCloud<pcl::PointXYZ>::Ptr &pts)
{
    //clear the dara in polar_bins_
    polar_bins_.clear();
    polar_bins_.resize(params_.num_bins_a*params_.num_bins_l);

    // gen polar bin grid
    float bsize_rad_inv = params_.num_bins_a/360.0;
    float bsize_lin_inv = params_.num_bins_l/params_.rmax;

    float rad_off = 360 * bsize_rad_inv;
    float rad_tr = 180.0 * bsize_rad_inv / M_PI;

    const int num_pts = pts->size();
    for (auto i=0; i<num_pts; ++i) {
        const auto &pt = pts->points[i];
        float tmp_r = sqrt(pt.x*pt.x + pt.y*pt.y);
        if(tmp_r > params_.rmax) continue;
        float tmp = atan2(pt.y, pt.x)*rad_tr;
        int rad_ind = tmp > 0 ? tmp : tmp+rad_off;
        rad_ind = rad_ind%params_.num_bins_a;
        //int rad_ind = static_cast<int>(rad_off + atan2(pt.y, pt.x) * rad_tr);
        int lin_ind = std::min(params_.num_bins_l-1, static_cast<int>(tmp_r * bsize_lin_inv));
        auto &current_pb = polar_bins_[rad_ind*params_.num_bins_l + lin_ind];
        if(current_pb.cell_pt_inds.size() == 0 || current_pb.height>pt.z)
        {
            current_pb.index = i;
            current_pb.height = pt.z;
            current_pb.r = tmp_r;
            current_pb.is_ground = false;
        }
        current_pb.cell_pt_inds.push_back(i);
    }
}

/**
 * @brief 根据距离和高度, 抽取sector ind的种子点, 并剔除掉那些点数很少(<5)的cell
 */
void gaussian_process_ground_seg::extract_seed(const int ind,
                                               const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts,
                                               std::vector<polar_bin_cell*> &current_model,
                                               std::vector<polar_bin_cell*> &sig_pts)
{
    current_model.clear();
    sig_pts.clear();
    const int start_ind = ind * params_.num_bins_l;
    const int end_ind = start_ind + params_.num_bins_l;

    for(int i=start_ind; i<end_ind-1; ++i)
    {
        if(polar_bins_[i].r > params_.max_seed_range) break;
        if(polar_bins_[i].cell_pt_inds.empty()) continue;
        // 防止太近
        const float cell_r = params_.rmax/params_.num_bins_l;
        int next_id = i+1;
        while(next_id<end_ind-1
               && (polar_bins_[next_id].cell_pt_inds.empty() || polar_bins_[next_id].r<polar_bins_[i].r+cell_r)) ++next_id;
        if(next_id == end_ind) break;
        float angle = atan2(polar_bins_[next_id].height - polar_bins_[i].height,
                            polar_bins_[next_id].r - polar_bins_[i].r);
        if(fabs(angle-params_.mount_angle)<0.3142) // 10 degree
        {
            polar_bins_[i].is_ground = true;
            polar_bins_[next_id].is_ground = polar_bins_[next_id].r<params_.max_seed_range;
        }
        i=next_id-1;
    }

    auto end_ptr = &polar_bins_[end_ind];
    for(auto ptr = &polar_bins_[start_ind]; ptr<end_ptr; ++ptr)
    {
        if(ptr->cell_pt_inds.empty()) continue;
        if(ptr->is_ground) {
            current_model.emplace_back(ptr);
        }
        else {
            sig_pts.emplace_back(ptr);
        }
    }


    // TODO optimize seed
//    for(int i=start_ind; i<end_ind; ++i)
//    {
//        auto &pbc = polar_bins_[i];
//        if(pbc.cell_pt_inds.size()<5) continue; // too few points in cell
//        if(pbc.r < params_.max_seed_range
//            && pbc.height < params_.max_seed_height) {
//            pbc.is_ground = true;
//            current_model.emplace_back(&pbc);
//        }
//        else sig_pts.emplace_back(&pbc);
//    }

    //std::cout << "rh pts:" << rh_pts_[ind].size() << std::endl;
    #if 0
    {
        std::stringstream ss;
        ss << "/home/wegatron/tmp/seed" << std::setw(4) << std::setfill('0') << ind << ".vtk";
        pcl::PointCloud<pcl::PointXYZ>::Ptr model_pc(new pcl::PointCloud<pcl::PointXYZ>);
        model_pc->resize(current_model.size());
        for(int i=0; i<current_model.size(); ++i)
            model_pc->points[i] = pts->points[current_model[i]->index];
        pcl::PointCloud<pcl::PointXYZ>::Ptr sig_pc(new pcl::PointCloud<pcl::PointXYZ>);
        sig_pc->resize(sig_pts.size());
        for(int i=0; i<sig_pts.size(); ++i) {
            sig_pc->points[i] = pts->points[sig_pts[i]->index];
        }
        zsw::point_clouds2vtk_file(ss.str(), {model_pc, sig_pc});
    }
    #endif

}

void gaussian_process_ground_seg::label_pc(const int ind,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr pts,
                                           std::vector<uint8_t> &labels)
{
    const int start_ind = ind * params_.num_bins_l;
    const int end_ind = start_ind + params_.num_bins_l;
    for(int i=start_ind; i<end_ind; ++i)
    {
        if(!polar_bins_[i].is_ground) continue;

        // segment ground points
        for(auto pt_ind : polar_bins_[i].cell_pt_inds)
        {
            if(pts->points[pt_ind].z < polar_bins_[i].height + params_.p_tg)
                labels[pt_ind] = 1;
        }
    }
}

std::vector<uint8_t> gaussian_process_ground_seg::segment(pcl::PointCloud<pcl::PointXYZ>::Ptr &pts)
{
    const auto num_pts = pts->size();
    std::vector<uint8_t> labels(num_pts, 0);
    if(num_pts <= 10) return labels;
    gen_polar_bin_grid(pts);
    std::vector<polar_bin_cell*> current_model;
    std::vector<polar_bin_cell*> sig_pts;
    for(int i=0; i<params_.num_bins_a; ++i) {
        std::cout << i << "/" << params_.num_bins_a << std::endl;
        extract_seed(i, pts, current_model, sig_pts);
        if(current_model.size() <= 2) continue;
        // insac
        insac(pts, 100, current_model, sig_pts);
        // result to labels
        label_pc(i, pts, labels);
//        for(const auto &tmp_cell : current_model)
//            labels[tmp_cell->index] = 1;
//        for(const auto &tmp_cell : sig_pts)
//            labels[tmp_cell->index] = 2;
    }
    #if 0
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr not_ground_pts(new pcl::PointCloud<pcl::PointXYZ>);

    for(int i=0; i<labels.size(); ++i) {
        if(labels[i] == 1) ground_pts->points.emplace_back(pts->points[i]);
        else if(labels[i] == 2) not_ground_pts->points.emplace_back(pts->points[i]);
    }
    zsw::point_clouds2vtk_file("/home/wegatron/tmp/segment.vtk", {ground_pts, not_ground_pts});
    #endif

#if 0
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_pts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr not_ground_pts(new pcl::PointCloud<pcl::PointXYZ>);

    for(int i=0; i<labels.size(); ++i) {
        if(labels[i] == 1) ground_pts->points.emplace_back(pts->points[i]);
        else not_ground_pts->points.emplace_back(pts->points[i]);
    }
    zsw::point_clouds2vtk_file("/home/wegatron/tmp/res.vtk", {ground_pts, not_ground_pts});
#endif
    return labels;
}

struct debug_info
{
    float r;
    float height;
    float predict;
    float cov;
};

void gaussian_process_ground_seg::insac(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts,
                                        const int max_iter,
                                        std::vector<polar_bin_cell*> &current_model,
                                        std::vector<polar_bin_cell*> &sig_pts)
{
    std::ofstream ofs("/home/wegatron/tmp/debug_insac.txt");
    // insac
    const float len_scale = 1.0/(2*params_.p_l*params_.p_l);
    int alter_cnt = 0;
    int itr_cnt = 0;
//    int tc_pre = 0;
//    int tc_val = 0;
//    int tc_evaluate = 0;
    do {
        //auto tp0 = std::chrono::high_resolution_clock::now();
        // calculate K_ff^{-1}, K_fy, K_yy
        const int model_size = current_model.size();
        const int sig_size = sig_pts.size();

        Eigen::VectorXf train_x(model_size);
        Eigen::VectorXf train_y(model_size);
        for(int j=0; j<model_size; ++j)
        {
            train_x[j] = current_model[j]->r;
            train_y[j] = current_model[j]->height;
        }

        Eigen::VectorXf x(sig_size);
        for(int j=0; j<sig_size; ++j)
            x[j] = sig_pts[j]->r;

        Eigen::VectorXf sqr_train_x = train_x.array() * train_x.array();
        Eigen::VectorXf sqr_x = x.array() * x.array();

        Eigen::MatrixXf K_ff(model_size, model_size);
        Eigen::MatrixXf inv_K_ff(model_size, model_size);
        {
            Eigen::MatrixXf k_0(model_size, model_size);
            Eigen::MatrixXf k_1(model_size, model_size);
            for(int j=0; j<model_size; ++j) {
                k_0.row(j).setConstant(sqr_train_x[j]);
                k_1.col(j).setConstant(sqr_train_x[j]);
            }

            K_ff = -len_scale * (k_0 + k_1 - 2*train_x*train_x.transpose());
            K_ff = params_.p_sf * exp(K_ff.array()).matrix()
                + params_.p_sn * Eigen::MatrixXf::Identity(model_size, model_size);
            inv_K_ff = K_ff.inverse();
        }

        Eigen::MatrixXf K_yy(sig_size, sig_size);
        {
            Eigen::MatrixXf k_0(sig_size, sig_size);
            Eigen::MatrixXf k_1(sig_size, sig_size);
            for(int j=0; j<sig_size; ++j) {
                k_0.row(j).setConstant(sqr_x[j]);
                k_1.col(j).setConstant(sqr_x[j]);
            }
            K_yy = -len_scale*(k_0 + k_1 - 2*x*x.transpose());
            K_yy = params_.p_sf * exp(K_yy.array());
        }

        Eigen::MatrixXf K_fy(model_size, sig_size);
        {
            Eigen::MatrixXf k_0(model_size, sig_size);
            Eigen::MatrixXf k_1(model_size, sig_size);
            for(int j=0; j<model_size; ++j)
                k_0.row(j).setConstant(sqr_train_x[j]);

            for(int j=0; j<sig_size; ++j)
                k_1.col(j).setConstant(sqr_x[j]);

            K_fy = -len_scale*(k_0 + k_1 - 2*train_x*x.transpose());
            K_fy = params_.p_sf * exp(K_fy.array());
        }

//        auto tp1 = std::chrono::high_resolution_clock::now();
//        tc_pre += std::chrono::duration_cast<std::chrono::milliseconds>(tp1 - tp0).count();

        // calc y and cov
        Eigen::MatrixXf tmp_calc = K_fy.transpose() * inv_K_ff;
        Eigen::VectorXf y = tmp_calc * train_y;
        Eigen::MatrixXf cov = K_yy - tmp_calc * K_fy;

//        auto tp2 = std::chrono::high_resolution_clock::now();
//        tc_val = std::chrono::duration_cast<std::chrono::milliseconds>(tp2-tp1).count();

#if 0
        {
            std::vector<debug_info> debug_cells(sig_size+model_size);
            for(int i=0; i<current_model.size(); ++i)
            {
                debug_cells[i] = {current_model[i]->r, current_model[i]->height, current_model[i]->height, 1.96f*sqrt(K_ff(i,i))};
                ofs << current_model[i]->r << " " << current_model[i]->height << " ";
            }

            ofs << std::endl;
            for(int i=0; i<sig_pts.size(); ++i)
            {
                debug_cells[i+model_size] ={sig_pts[i]->r, sig_pts[i]->height, y[i], 1.96f*sqrt(cov(i,i))};
            }
            std::sort(debug_cells.begin(), debug_cells.end(), [](debug_info &a, debug_info &b){ return a.r < b.r;});

            for(int i=0; i<debug_cells.size(); ++i)
            {
                ofs << debug_cells[i].r << " " << debug_cells[i].height
                    << " " << debug_cells[i].predict << " " << debug_cells[i].cov << " ";
            }

            ofs << std::endl;
        }
#endif

        // evaluate the model
        alter_cnt = 0;
        for(int i=0; i<sig_size; ++i)
        {
            float cur_sqr_sigma_inv = 1.0/cov(i,i);
            float cur_sigma_inv = 0.3984*sqrt(cur_sqr_sigma_inv);
            float p_0 = cur_sigma_inv * exp(-0.5 * (sig_pts[i]->height - y[i]-params_.p_tdata)*cur_sqr_sigma_inv);
            float p_1 = cur_sigma_inv * exp(-0.5 *(sig_pts[i]->height - y[i]+params_.p_tdata)*cur_sqr_sigma_inv);

            if(y[i] < params_.p_tmodel // y[i] seems fine
                && (p_0+p_1)*params_.p_tdata*0.5>0.8)
            {
                current_model.push_back(sig_pts[i]);
                sig_pts[i]->is_ground = true;
                sig_pts[i] = sig_pts[sig_pts.size()-1];
                sig_pts.pop_back();
                ++alter_cnt;
            }
        }

//        auto tp3 = std::chrono::high_resolution_clock::now();
//        tc_evaluate = std::chrono::duration_cast<std::chrono::milliseconds>(tp3-tp2).count();
        // std::cout << "y:" << y.transpose() << std::endl;
        // std::cout << "cov:\n" << cov << std::endl;
//        std::cout << "alter_cnt:" << alter_cnt << std::endl;
//        std::cout << "model_pts_num=" << current_model.size()
//                  << " sig_pts_num=" << sig_pts.size() << std::endl;
        // debug output
#if 0
        {
            std::stringstream ss;
            ss << "/home/wegatron/tmp/insac_"
               << std::setw(4) << std::setfill('0') << itr_cnt << ".vtk";
            pcl::PointCloud<pcl::PointXYZ>::Ptr model_pc(new pcl::PointCloud<pcl::PointXYZ>);
            model_pc->resize(current_model.size());
            for(int i=0; i<current_model.size(); ++i)
            {
                model_pc->points[i] = pts->points[current_model[i]->index];
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr sig_pc(new pcl::PointCloud<pcl::PointXYZ>);
            sig_pc->resize(sig_pts.size());
            for(int i=0; i<sig_pts.size(); ++i)
            {
                sig_pc->points[i] = pts->points[sig_pts[i]->index];
            }
            zsw::point_clouds2vtk_file(ss.str(), {model_pc, sig_pc});
        }
#endif
    }while(alter_cnt >0 && ++itr_cnt < max_iter);
//    std::cout << "tc_pre=" << tc_pre << " tc_val=" << tc_val << " tc_evaluate=" << tc_evaluate << std::endl;
    std::cout << "itr " << itr_cnt << std::endl;
}

boost::python::numpy::ndarray gaussian_process_ground_seg::segment_py(const boost::python::numpy::ndarray &pts_data)
{
    const auto pt_num = pts_data.shape(0);
    float * data_ptr = reinterpret_cast<float*>(pts_data.get_data());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts(new pcl::PointCloud<pcl::PointXYZ>);
    pts->resize(pt_num);
    for(auto i=0; i<pt_num; ++i)
    {
        pts->points[i].x = *data_ptr; ++data_ptr;
        pts->points[i].y = *data_ptr; ++data_ptr;
        pts->points[i].z = *data_ptr; ++data_ptr;
    }
    auto labels = segment(pts);
    np::ndarray ret = np::empty(boost::python::make_tuple(pt_num), np::dtype::get_builtin<uint8_t>());
    memcpy(ret.get_data(), labels.data(), pt_num);
    return ret;
}

BOOST_PYTHON_MODULE(wegatron_ground_seg) {
    using namespace boost::python;
    np::initialize(); // must initialize before using numpy
    class_<gaussian_process_ground_seg>("gaussian_process_ground_seg",
                                        init<const std::string &>())
        .def("segment", &gaussian_process_ground_seg::segment_py);
}
