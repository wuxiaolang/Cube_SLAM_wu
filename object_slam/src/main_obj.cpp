#include <iostream>
#include <fstream>
#include <string> 
#include <sstream>
#include <ctime>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <ros/ros.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"


#include <object_slam/Object_landmark.h>
#include <object_slam/Frame.h>
#include <object_slam/g2o_Object.h>

#include "detect_3d_cuboid/matrix_utils.h"
#include "detect_3d_cuboid/detect_3d_cuboid.h"

#include "line_lbd/line_lbd_allclass.h"

using namespace std;
using namespace Eigen;

typedef pcl::PointCloud<pcl::PointXYZRGB> CloudXYZRGB;

// global variable
std::string base_folder;
bool online_detect_mode;
bool save_results_to_txt;
cv::Mat_<float> matx_to3d_, maty_to3d_;

// BRIEF 深度图转化为点云图参数.
void set_up_calibration(const Eigen::Matrix3f& calibration_mat,const int im_height,const int im_width)
{
    matx_to3d_.create(im_height, im_width);
    maty_to3d_.create(im_height, im_width);
    float center_x=calibration_mat(0,2);  //cx
    float center_y=calibration_mat(1,2);  //cy
    float fx_inv=1.0/calibration_mat(0,0);  // 1/fx
    float fy_inv=1.0/calibration_mat(1,1);  // 1/fy
    for (int v = 0; v < im_height; v++) {
	for (int u = 0; u < im_width; u++) {
	  matx_to3d_(v,u) = (u - center_x) * fx_inv;
	  maty_to3d_(v,u) = (v - center_y) * fy_inv;
	}
    }
}

// BRIEF 下采样点云图，深度图已经以 m 为单位.
// 输入：原始rgb图 rgb_img		深度图 depth_img 		相机真实位姿 transToWorld		采样之后的点云 point_cloud
void depth_to_cloud(const cv::Mat& rgb_img, const cv::Mat& depth_img,const Eigen::Matrix4f transToWorld, CloudXYZRGB::Ptr& point_cloud,bool downsample=false)
{
    pcl::PointXYZRGB pt;
    pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> vox_grid_;
    float close_depth_thre = 0.1;
    float far_depth_thre = 3.0;
      far_depth_thre = 3;
    int im_width = rgb_img.cols; int im_height= rgb_img.rows;
    for (int32_t i=0; i<im_width*im_height; i++) 
	{      // row by row
		int ux=i % im_width; int uy=i / im_width;       
		float pix_depth= depth_img.at<float>(uy,ux);
		if (pix_depth>close_depth_thre && pix_depth<far_depth_thre)
		{
			pt.z=pix_depth; pt.x=matx_to3d_(uy,ux)*pix_depth; pt.y=maty_to3d_(uy,ux)*pix_depth;
			Eigen::VectorXf global_pt=homo_to_real_coord_vec<float>(transToWorld*Eigen::Vector4f(pt.x,pt.y,pt.z,1));  // change to global position
			pt.x=global_pt(0); pt.y=global_pt(1); pt.z=global_pt(2);
			pt.r = rgb_img.at<cv::Vec3b>(uy,ux)[2]; pt.g = rgb_img.at<cv::Vec3b>(uy,ux)[1]; pt.b = rgb_img.at<cv::Vec3b>(uy,ux)[0];
			point_cloud->points.push_back(pt);
		}
    }    
    if (downsample)
    {
		vox_grid_.setLeafSize(0.02,0.02,0.02);
		vox_grid_.setDownsampleAllData(true);
		vox_grid_.setInputCloud(point_cloud);
		vox_grid_.filter(*point_cloud);
    }
}

// BRIEF 前后标记立方体.
// one cuboid need front and back markers...
// 输入参数： 8个顶点		marker		id
void cuboid_corner_to_marker(const Matrix38d& cube_corners,visualization_msgs::Marker& marker, int bodyOrfront)
{
    Eigen::VectorXd edge_pt_ids;
	// NOTE 整体的边缘.
    if (bodyOrfront==0) 
	{
		edge_pt_ids.resize(16); 
		// NOTE 按顺序连接各个点，连成一个立方体
		edge_pt_ids << 1,2,3,4,1,5,6,7,8,5,6,2,3,7,8,4;
		edge_pt_ids.array()-=1;
    }
	// NOTE 正前方的边缘.
	else 
	{
		edge_pt_ids.resize(5); 
		// NOTE 按顺序连接各个点，平面
		edge_pt_ids<<1,2,6,5,1;
		edge_pt_ids.array()-=1;
    }

    marker.points.resize(edge_pt_ids.rows());
	// 确定各个点的位置，连成线.
    for (int pt_id = 0; pt_id < edge_pt_ids.rows(); pt_id++)
    {
		marker.points[pt_id].x = cube_corners(0, edge_pt_ids(pt_id));
		marker.points[pt_id].y = cube_corners(1, edge_pt_ids(pt_id));
		marker.points[pt_id].z = cube_corners(2, edge_pt_ids(pt_id));
    }
}

// BRIEF 立方体需要有前后标记，立方体模型的表达.
// 输入参数：立方体提案  颜色.
// one cuboid need front and back markers...  rgbcolor is 0-1 based
visualization_msgs::MarkerArray cuboids_to_marker(object_landmark* obj_landmark, Vector3d rgbcolor) 
{
    visualization_msgs::MarkerArray plane_markers;  
	visualization_msgs::Marker marker;

    if (obj_landmark==nullptr)
		return plane_markers;

	// 设置 marker.
    marker.header.frame_id="/world";  
	marker.header.stamp=ros::Time::now();
    marker.id = 0; //0
    marker.type = visualization_msgs::Marker::LINE_STRIP;   
	marker.action = visualization_msgs::Marker::ADD;
    marker.color.r = rgbcolor(0); 
	marker.color.g = rgbcolor(1); 
	marker.color.b = rgbcolor(2); 
	marker.color.a = 1.0;
    marker.scale.x = 0.02;

	// 显示的立方体对象 cube_opti.
    g2o::cuboid cube_opti = obj_landmark->cube_vertex->estimate();
	// 立方体 8 个顶点的坐标.
    Eigen::MatrixXd cube_corners = cube_opti.compute3D_BoxCorner();
    // std::cout << "cube_corners:   \n" << cube_corners << std::endl;

	// 每个立方体需要两个 marker，一个用于所有的边缘，一个用于前面的边缘，可以具有不同的额颜色.
    for (int ii=0;ii<2;ii++) // each cuboid needs two markers!!! one for all edges, one for front facing edge, could with different color.
    {
		marker.id++;
		cuboid_corner_to_marker(cube_corners,marker, ii);
		plane_markers.markers.push_back(marker);
    }
    return plane_markers;
}

// BRIEF 将李代数表达的相机位姿 pose_Twc 转换成 geometry_msgs/Pose 消息.
geometry_msgs::Pose posenode_to_geomsgs(const g2o::SE3Quat &pose_Twc)
{
    geometry_msgs::Pose pose_msg;    
    Eigen::Vector3d pose_trans = pose_Twc.translation();	
    pose_msg.position.x=pose_trans(0);
    pose_msg.position.y=pose_trans(1);
    pose_msg.position.z=pose_trans(2);
    Eigen::Quaterniond pose_quat = pose_Twc.rotation();
    pose_msg.orientation.x = pose_quat.x();  
    pose_msg.orientation.y = pose_quat.y();
    pose_msg.orientation.z = pose_quat.z();
    pose_msg.orientation.w = pose_quat.w();
    return pose_msg;
}

// BRIEF 将李代数表达的相机位姿 pose_Twc 加上标签 img_header 之后转换成 nav_msgs/Odometry 消息.
nav_msgs::Odometry posenode_to_odommsgs(const g2o::SE3Quat &pose_Twc,const std_msgs::Header &img_header)
{
    nav_msgs::Odometry odom_msg;
    odom_msg.pose.pose=posenode_to_geomsgs(pose_Twc);    
    odom_msg.header=img_header;
    return odom_msg;
}

// BRIEF 发布每帧的原始和优化结果.
void publish_all_poses(std::vector<tracking_frame*> all_frames,std::vector<object_landmark*> cube_landmarks_history,
		       std::vector<object_landmark*> all_frame_rawcubes, Eigen::MatrixXd& truth_frame_poses)
{
	// STEP 1.定义 ROS 消息发布器和相关变量.
    ros::NodeHandle n;
	// 估计的和真实的相机【运动轨迹】
	ros::Publisher pub_slam_path = n.advertise<nav_msgs::Path>( "/slam_pose_paths", 10 );
    ros::Publisher pub_truth_path = n.advertise<nav_msgs::Path>( "/truth_pose_paths", 10 );
	// 估计和真实的【相机位姿】（一次性显示）.
    ros::Publisher pub_slam_all_poses = n.advertise<geometry_msgs::PoseArray>("/slam_pose_array", 10);
    ros::Publisher pub_truth_all_poses = n.advertise<geometry_msgs::PoseArray>("/truth_pose_array", 10);
	// 估计的和真实【相机位姿】(逐帧显示).
    ros::Publisher pub_slam_odompose = n.advertise<nav_msgs::Odometry>("/slam_odom_pose", 10);
    ros::Publisher pub_truth_odompose = n.advertise<nav_msgs::Odometry>("/truth_odom_pose", 10);
	// 【立方体模型】消息，分别是：最终路标，优化后模型，优化前模型.
    ros::Publisher pub_final_opti_cube = n.advertise<visualization_msgs::MarkerArray>("/cubes_opti", 10); //最终估计出的立方体提案.
    ros::Publisher pub_history_opti_cube = n.advertise<visualization_msgs::MarkerArray>("/cubes_opti_hist", 10); // 每次优化后的立方体路标
    ros::Publisher pub_frame_raw_cube = n.advertise<visualization_msgs::MarkerArray>("/cubes_raw_frame", 10);
    // 带有提案的【原始图像】.
	ros::Publisher pub_2d_cuboid_project = n.advertise<sensor_msgs::Image>("/cuboid_project_img", 10);
	// 【点云】消息.
    ros::Publisher raw_cloud_pub = n.advertise<CloudXYZRGB> ("/raw_point_cloud", 50);
    
    int total_frame_number = all_frames.size();
    
    // 估计的【相机位姿】.
    geometry_msgs::PoseArray all_pred_pose_array;    		// 估计位姿，存为数组，一次行发布所有.
	std::vector<nav_msgs::Odometry> all_pred_pose_odoms;	// 估计位姿，存为向量，逐帧发布.

	// 真实的【相机位姿】.
    geometry_msgs::PoseArray all_truth_pose_array;    		// 真实位姿，存为数组，一次性发布.
	std::vector<nav_msgs::Odometry> all_truth_pose_odoms;	// 真实位姿，存为向量，逐帧发布.

	// pose_header：指明在世界坐标系和时间戳.
    std_msgs::Header pose_header;    				// std_msgs/Header：通常用于在特定坐标系中传送带时间戳的数据.
	pose_header.frame_id = "/world";    
	pose_header.stamp = ros::Time::now();

	// 【运动轨迹】，包括位姿和时间戳：path_preds，path_truths.
    nav_msgs::Path path_truths,path_preds;			// nav_msgs/Path：姿态数组，表示运动轨迹.
    path_preds.header = pose_header;    
	path_truths.header = pose_header;    

	// STEP 2.读取估计的优化之后的相机位姿.
    for (int i = 0; i < total_frame_number; i++)
    {
		// cam_pose_Twc：优化之后的相机位姿.
		// 保存优化之后的相机位姿和时间戳.
		// 分别保存为数组（一次性发布）和向量形式（逐帧发布）.
		all_pred_pose_array.poses.push_back(posenode_to_geomsgs(all_frames[i]->cam_pose_Twc));
		all_pred_pose_odoms.push_back(posenode_to_odommsgs(all_frames[i]->cam_pose_Twc,pose_header) );	

		/*NOTE 作者原始代码在这里将估计的相机位姿存放在 path_preds 中，并在后面一次性发布，我给他放到后面逐帧发布了.
		// 位姿时间戳 postamp.
		geometry_msgs::PoseStamped postamp;
		postamp.pose = posenode_to_geomsgs(all_frames[i]->cam_pose_Twc);
		postamp.header = pose_header;
		// NOTE 最终将估计的位姿信息存储到 path_preds.
		path_preds.poses.push_back(postamp);
		*/
    }

	// STEP 3.读取真实的相机位姿..
    if (truth_frame_poses.rows()>0)
    {
		for (int i=0; i < total_frame_number;i++)
		{
			// 在自由空间中的位姿表示，包括位置和方向.
			geometry_msgs::Pose pose_msg;
			pose_msg.position.x=truth_frame_poses(i,1);    
			pose_msg.position.y=truth_frame_poses(i,2);    
			pose_msg.position.z=truth_frame_poses(i,3);

			pose_msg.orientation.x = truth_frame_poses(i,4);	
			pose_msg.orientation.y = truth_frame_poses(i,5);
			pose_msg.orientation.z = truth_frame_poses(i,6);	
			pose_msg.orientation.w = truth_frame_poses(i,7);

			// 真实位姿矩阵.
			all_truth_pose_array.poses.push_back(pose_msg);

			// 真实位姿的里程计信息（向量）.
			nav_msgs::Odometry odom_msg;
			odom_msg.pose.pose=pose_msg;
			odom_msg.header = pose_header;
			all_truth_pose_odoms.push_back(odom_msg);
			
			/*NOTE 作者原始代码在这里将真实的相机位姿存放在 path_truths 中，并在后面一次性发布，我给他放到后面逐帧发布了.
			geometry_msgs::PoseStamped postamp;
			postamp.pose = pose_msg;
			postamp.header = pose_header;
			// NOTE 最终将参考系下真实的位姿信息存储到 path_truths.
			path_truths.poses.push_back(postamp);
			*/
		}
    }

	// 指定坐标系和时间戳.
    all_pred_pose_array.header.stamp=ros::Time::now();    
	all_pred_pose_array.header.frame_id="/world";
    all_truth_pose_array.header.stamp=ros::Time::now();    
	all_truth_pose_array.header.frame_id="/world";
    
    // STEP 4.保存相机和物体的位姿.
    if (save_results_to_txt)  // record cam pose and object pose
    {
		// 写入相机位姿：时间戳、位置、方向.
		ofstream resultsFile;
		string resultsPath = base_folder + "output_cam_poses.txt";
		cout << "resultsPath  " << resultsPath << endl;
		resultsFile.open(resultsPath.c_str());
		resultsFile << "# timestamp tx ty tz qx qy qz qw"<<"\n";
		for (int i=0; i<total_frame_number; i++)
		{
			// 时间戳读取真实的相机的时间戳.
			double time_string=truth_frame_poses(i,0);
			ros::Time time_img(time_string);
			resultsFile << time_img<<"  ";	    
			// 读取位置信息，转换为向量形式.
			resultsFile << all_frames[i]->cam_pose_Twc.toVector().transpose()<<"\n";
		}
		resultsFile.close();
		
		// 写入物体位姿.
		ofstream objresultsFile;
		string objresultsPath = base_folder + "output_obj_poses.txt";
		objresultsFile.open(objresultsPath.c_str());
		for (size_t j=0;j<cube_landmarks_history.size();j++)
		{
			// 优化之后的相机位姿 cube_landmarks_history.
			g2o::cuboid cube_opti = cube_landmarks_history[j]->cube_vertex->estimate();
			// transform it to local ground plane.... suitable for matlab processing.
			objresultsFile << cube_opti.toMinimalVector().transpose()<<" "<<"\n";
		}
		objresultsFile.close();
    }
    
	// 数据集的相机参数.
    // sensor parameter for TUM cabinet data!
    Eigen::Matrix3f calib;
    float depth_map_scaling = 5000;
    calib<<535.4,      0, 320.1,
	           0,  539.2, 247.6,
	           0,      0,     1;
    set_up_calibration(calib,480,640);
    
	// STEP 5. 最终的立方体路标 finalcube_markers.
	// 传入参数：优化后提案 cube_landmarks_history 向量的最后一个元素，颜色绿色.
	// TODO finalcube_markers
    visualization_msgs::MarkerArray finalcube_markers = cuboids_to_marker(cube_landmarks_history.back(),Vector3d(0,1,0));
    
	// 是否显示点云信息，用于可视化.
    bool show_truth_cloud = true;  // show point cloud using camera pose. for visualization purpose
        
    pcl::PCLPointCloud2 pcd_cloud2;		// 好像没用到.
    
    ros::Rate loop_rate(5);  //5
    int frame_number = -1;

	// STEP 6. 发布每一帧对应的消息.
    while ( n.ok() )
    {
		frame_number++;
	
		// 是否直接一次性显示所有结果.
		if (0) // directly show final results
		{
			// 发布估计和真实的相机位姿.
			pub_slam_all_poses.publish(all_pred_pose_array);	
			pub_truth_all_poses.publish(all_truth_pose_array);
			// 发布估计和真实的运动轨迹.
			pub_slam_path.publish(path_preds);	
			pub_truth_path.publish(path_truths);
		}

		// STEP 6.1 发布最终确定的立方体模型.
		pub_final_opti_cube.publish(finalcube_markers);
	
		if (frame_number < total_frame_number)
		{
			// STEP 6.2 发布立方体位姿（优化前后）信息.
			// NOTE 在每帧经过G20优化之后发布其立方体路标，用红色表示 pub_history_opti_cube. 
			if (cube_landmarks_history[frame_number]!=nullptr)
				pub_history_opti_cube.publish(cuboids_to_marker(cube_landmarks_history[frame_number],Vector3d(1, 0, 0)));
			// NOTE 发布优化之前每帧中原始的立方体，用蓝色表示 pub_frame_raw_cube.
			if (all_frame_rawcubes.size()>0 && all_frame_rawcubes[frame_number]!=nullptr)
				pub_frame_raw_cube.publish(cuboids_to_marker(all_frame_rawcubes[frame_number],Vector3d(0,0,1)));

			// STEP 6.3 发布相机位姿（估计和真实）信息.
			// NOTE 发布该帧对应的位姿信息.
			// 估计的位姿：pub_slam_odompose.
			// 真实的位姿：pub_truth_odompose.
			pub_slam_odompose.publish(all_pred_pose_odoms[frame_number]);
			pub_truth_odompose.publish(all_truth_pose_odoms[frame_number]);
			// 输出相机在世界中的坐标.
		    std::cout<<"Frame position x/y/z:  "<< frame_number <<"        "
			                                    << all_pred_pose_odoms[frame_number].pose.pose.position.x << "  "
											    << all_pred_pose_odoms[frame_number].pose.pose.position.y << "  "
											    << all_pred_pose_odoms[frame_number].pose.pose.position.z << std::endl;

			// 将 frame_number 化为4位字符 frame_index_c
			char frame_index_c[256];	
			sprintf(frame_index_c,"%04d",frame_number);  // format into 4 digit
			
			// 读取该帧带有立方体提案的图像.
			cv::Mat cuboid_2d_proj_img = all_frames[frame_number]->cuboids_2d_img;

			// 读取原始 rgb 图像 raw_rgb_img.
			std::string raw_rgb_img_name = base_folder+"raw_imgs/" + std::string(frame_index_c) + "_rgb_raw.jpg";
			cv::Mat raw_rgb_img = cv::imread(raw_rgb_img_name, 1);

			// STEP 6.4 发布点云信息.
			// 隔帧显示点云图.
			if (show_truth_cloud && (truth_frame_poses.rows()>0))
			{
				if (frame_number%4==0) // show point cloud every N frames
				{
					// 读取点云图 raw_depth_img 并转换格式.
					std::string raw_depth_img_name = base_folder+"depth_imgs/" + std::string(frame_index_c) + "_depth_raw.png";
					cv::Mat raw_depth_img = cv::imread(raw_depth_img_name, CV_LOAD_IMAGE_ANYDEPTH);
					raw_depth_img.convertTo(raw_depth_img, CV_32FC1, 1.0/depth_map_scaling,0);

					// 读取点云信息 point_cloud.
					CloudXYZRGB::Ptr point_cloud(new CloudXYZRGB());

					// 读取真实的相机位姿，李代数表示
					Eigen::Matrix4f truth_pose_matrix=g2o::SE3Quat(truth_frame_poses.row(frame_number).segment<7>(1)).to_homogeneous_matrix().cast<float>();
					// std::cout<<"truth_pose_matrix:   \n"<< truth_pose_matrix << std::endl;

					// 下采样点云，否则太多.
					depth_to_cloud(raw_rgb_img, raw_depth_img, truth_pose_matrix, point_cloud, true); // need to downsample cloud, otherwise too many
					ros::Time curr_time=ros::Time::now();

					// 发布点云图消息.
					point_cloud->header.frame_id = "/world";
					point_cloud->header.stamp = (curr_time.toNSec() / 1000ull);
					raw_cloud_pub.publish(point_cloud);
				}
			}
			
			// STEP 6.5 发布原始图与提案消息.
			// 发布带有立方体轮廓的图像，通过 pub_2d_cuboid_project 发布.
			cv_bridge::CvImage out_image;
			out_image.header.stamp=ros::Time::now();
			out_image.image=cuboid_2d_proj_img;
			out_image.encoding=sensor_msgs::image_encodings::TYPE_8UC3;
			pub_2d_cuboid_project.publish(out_image.toImageMsg());

			// TODO 逐帧发布估计的轨迹 path_preds.
			geometry_msgs::PoseStamped postamp;
			postamp.pose = posenode_to_geomsgs(all_frames[frame_number]->cam_pose_Twc);
			postamp.header = pose_header;
			path_preds.poses.push_back(postamp);
			// 发布.
			pub_slam_path.publish(path_preds);

			// TODO 逐帧发布真实的轨迹 path_truths.
			geometry_msgs::PoseStamped postamp2;
			postamp2.pose = all_truth_pose_array.poses[frame_number];
			postamp2.header = pose_header;
			path_truths.poses.push_back(postamp2);
			// 发布.
			pub_truth_path.publish(path_truths);
		}
		
		// 结束可视化阶段.
		if (frame_number==int(all_pred_pose_odoms.size()))
		{
			cout<<"+++++++++++++Finish all visulialization!+++++++++++++"<<endl;
		}
		
		ros::spinOnce();
		loop_rate.sleep();
    }  
}

// NOTE 在线检测模式不使用 pred_frame_objects 和 init_frame_poses.
//     truth_frame_poses 仅使用第一帧.
void incremental_build_graph(Eigen::MatrixXd& offline_pred_frame_objects, Eigen::MatrixXd& init_frame_poses, Eigen::MatrixXd& truth_frame_poses)
{
	// STEP 【1. 变量定义】
    // 设置 TUM 数据集相机内参calib.
    Eigen::Matrix3d calib; 
    calib<<535.4,  0,  320.1,   
	    0,  539.2, 247.6,
	    0,      0,     1;   

    // 帧数，也即truth_frame_poses中的列数.
    int total_frame_number = truth_frame_poses.rows();

	// STEP 【1.2 定义立方体检测对象】
    // 检测所有帧中的立方体，定义为一个【detect_3d_cuboid】类的detect_cuboid_obj对象.
    detect_3d_cuboid detect_cuboid_obj;
    detect_cuboid_obj.whether_plot_detail_images = false;	// 不绘制检测细节图像.
    detect_cuboid_obj.whether_plot_final_images = false;	// 不绘制检测结果图.
    detect_cuboid_obj.print_details = false;  				// 不输出检测细节.
    detect_cuboid_obj.set_calibration(calib);				// 设置内参.
    detect_cuboid_obj.whether_sample_bbox_height = false;	// TODO
    detect_cuboid_obj.nominal_skew_ratio = 2;				// TODO
    detect_cuboid_obj.whether_save_final_images = true;		// 保存检测结果图.
	// STEP 【1.3 定义线检测对象】
    // 定义线检测【line_lbd_detect】类的line_lbd_obj对象.
    line_lbd_detect line_lbd_obj;
    line_lbd_obj.use_LSD = false;		// 使用 LSD 或 detector 线检测.
    line_lbd_obj.line_length_thres = 15;  	// 去除较短的边线.
    
    // STEP 【2.G2O图优化 graph optimization】
    //in this example, there is only one object!!! perfect association  假设只有一个对象！！！  TODO
    // STEP 【2.1 构造一个名为 graph 的求解器】
    g2o::SparseOptimizer graph;
    // STEP 【2.2 使用Cholmod中的线性方程求解器得到 linearSolver】
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    // STEP 【2.3 再用稀疏矩阵块求解器 solver_ptr】
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    // STEP 【2.4使用梯度下降算法求解上面的 solver_ptr 得到 solver】
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
    graph.setAlgorithm(solver);		// 设置求解器.    
    graph.setVerbose(false);		// 是否打开调试输出.

    
    // only first truth pose is used. to directly visually compare with truth pose. also provide good roll/pitch
    // 这里仅使用了第一帧的相机真实位姿，为了直接与真实位姿对比，也提供良好的 roll/pitch 角.
	// 构造一个四元数形式的 fixed_init_cam_pose_Twc 读取第一帧的位姿.
    g2o::SE3Quat fixed_init_cam_pose_Twc(truth_frame_poses.row(0).tail<7>());
    
    // 保存每帧的优化结果.
	// 每帧优化后的路标的位姿.
    std::vector<object_landmark*> cube_pose_opti_history(total_frame_number, nullptr);  //landmark pose after each frame's optimization
    // 优化之前每帧检测到的立方体帧.
	std::vector<object_landmark*> cube_pose_raw_detected_history(total_frame_number, nullptr); //raw detected cuboid frame each frame. before optimization

    int offline_cube_obs_row_id = 0;
    
	// @PARAM all_frames 存储每一帧的信息.
    std::vector<tracking_frame*> all_frames(total_frame_number);    // 一个 tracking_frame 向量.
    g2o::VertexCuboid* vCube;		// TODO 这个顶点将物体的位姿存储到世界.
    
    // 在线逐帧处理每一帧图像.
    for (int frame_index=0; frame_index<total_frame_number; frame_index++)
    { 
		// @PARAM 李代数位姿表示.
		g2o::SE3Quat curr_cam_pose_Twc;		// 当前帧的位姿.
		g2o::SE3Quat odom_val; 				// 从上一帧到当前帧的旋转.
		
		// STEP 1. 计算每一帧的位姿.
		if (frame_index==0)
			curr_cam_pose_Twc = fixed_init_cam_pose_Twc;		// 读取第一帧的真实位姿.
		else
		{
			// @PARAM 上一帧的位姿：all_frames[frame_index-1] 的 Tcw.
			g2o::SE3Quat prev_pose_Tcw = all_frames[frame_index-1]->cam_pose_Tcw;
			// 从第三帧开始，使用恒定运动模型来初始化相机.
			// TODO 第 0 帧来自于真实位姿，第 2 帧通过里程计计算出来，第 1 帧呢？？
			if (frame_index > 1)  // from third frame, use constant motion model to initialize camera.
			{
				// 上上一帧的位姿 prev_prev_pose_Tcw.
				g2o::SE3Quat prev_prev_pose_Tcw = all_frames[frame_index-2]->cam_pose_Tcw;
				// 从上一帧到本帧的变换 
				odom_val = prev_pose_Tcw*prev_prev_pose_Tcw.inverse();
			}
			// NOTE 计算当前帧的位姿（相机->相机） CT_wc =(T_odom)^-1
			curr_cam_pose_Twc = (odom_val*prev_pose_Tcw).inverse();
		}
		// std::cout << "第 " << frame_index << "帧的位姿：\n" << curr_cam_pose_Twc << std::endl;
		// std::cout << "odom_val：\n" << odom_val << std::endl;
	  
	  // STEP 2. 信息存储.
	  // 将当前帧 currframe 的信息存放在 all_frames 序列中.
	  tracking_frame* currframe = new tracking_frame();
	  currframe->frame_seq_id = frame_index;
	  all_frames[frame_index] = currframe;
	  
	  bool has_detected_cuboid = false;
	  // @PARAM 定义一个立方体类的对象 cube_local_meas
	  g2o::cuboid cube_local_meas; 
	  double proposal_error;
	  char frame_index_c[256];	
	  // 将当前帧的编号写入到 frame_index_c 中. TODO 这里又将frame_index写入到frame_index_c做什么？仅仅格式化位4位数？
	  sprintf(frame_index_c,"%04d",frame_index);  // 格式化为4位.

	  // read or detect cuboid
	  // STEP 3. 读取（离线）或【检测（在线）立方体物体】.
	  if (online_detect_mode)
	  {
	      // STEP 3.1 开始检测，读取rgb图像到 raw_rgb_img.
	      cv::Mat raw_rgb_img = cv::imread(base_folder+"raw_imgs/"+frame_index_c+"_rgb_raw.jpg", 1);
	    
	      // STEP 3.2 【边缘线检测】.
		  // @PARAM all_lines_raw 边缘线存储的矩阵.
	      cv::Mat all_lines_mat;	// 检测到的线段信息，cv::Mat格式.
	      line_lbd_obj.detect_filter_lines(raw_rgb_img, all_lines_mat);

		  // 将 all_lines_mat 存储到 all_lines_raw 中.
	      Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);		// TODO这里的4是什么，4条线吗？？
	      for (int rr=0;rr<all_lines_mat.rows;rr++)
				for (int cc=0;cc<4;cc++)
		  			all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);
					/*
					std::cout << "all_lines_raw \n" << all_lines_raw << std::endl;
					518.164  179.13     533      46
					453.637 371.208   516.2 180.066
					285.62 322.261 451.626 372.243
					290.387 120.984 285.264 319.981
					384.164 22.1538 291.708 120.726
					514.869 171.607 290.926 123.344
					380.963  167.12 398.815 172.601
					381.094 202.619 395.935 206.264
					397.868 176.387 379.907 170.274
					*/

		  // STEP 3.3 读取 yolo 2D 目标检测.
	      //read cleaned yolo 2d object detection
	      Eigen::MatrixXd raw_2d_objs(10,5);  // 每帧 5 个参数：2D检测框[x1 y1 width height], 和概率. TODO 
	      if (!read_all_number_txt(base_folder+"/filter_2d_obj_txts/"+frame_index_c+"_yolo2_0.15.txt", raw_2d_objs))
		  		return;
		//   else 
		//   		std::cout << "yolo:" << "\n" << raw_2d_objs << std::endl;
	      raw_2d_objs.leftCols<2>().array() -=1;   // 将matlab的坐标换成c++，减1；change matlab coordinate to c++, minus 1
	      
		  // STEP 3.4 将当前帧的李代数表示的位姿转换成Eigen的变换矩阵.
	      Matrix4d transToWolrd;
	      detect_cuboid_obj.whether_sample_cam_roll_pitch = (frame_index!=0); // 第一帧可以不用采样相机位姿，也可以采样，不重要；first frame doesn't need to sample cam pose. could also sample. doesn't matter much
	      if (detect_cuboid_obj.whether_sample_cam_roll_pitch) //sample around first frame's pose
		  		transToWolrd = fixed_init_cam_pose_Twc.to_homogeneous_matrix();		// 将李代数表示的变换转换成Eigen矩阵表示transToWolrd.
	      else
		  transToWolrd = curr_cam_pose_Twc.to_homogeneous_matrix();
	      
		  // STEP 3.5 立方体检测
	      std::vector<ObjectSet> frames_cuboids; // 立方体提案的排序向量. ObjectSet 是一个matlab的立方体对象.
		  // NOTE detect_cuboid()函数检测立方体.
	      detect_cuboid_obj.detect_cuboid(raw_rgb_img,transToWolrd,raw_2d_objs,all_lines_raw, frames_cuboids);
	      // cuboids_2d_img：带有立方体提案的 2D 图像.
		  currframe->cuboids_2d_img = detect_cuboid_obj.cuboids_2d_img;
		  cv::imshow("currframe.cuboids_2d_img",currframe->cuboids_2d_img);
		  cv::waitKey(0);

		  // STEP 3.6 检测到了物体，准备物体测量.
	      has_detected_cuboid = frames_cuboids.size()>0 && frames_cuboids[0].size()>0;		// frames_cuboids[0] 第0个对象.
	      if (has_detected_cuboid)  // prepare object measurement
	      {
			  	// 第 0 个物体的第 0 个提案.
				cuboid* detected_cube = frames_cuboids[0][0];  // NOTE this is a simple dataset, only one landmark

				g2o::cuboid cube_ground_value; // local ground frame 中的立方体对象.
				// 立方体 9 自由度的位姿.
				Vector9d cube_pose;
				cube_pose  << 	detected_cube->pos(0),
								detected_cube->pos(1),
								detected_cube->pos(2),
								0,
								0,
								detected_cube->rotY,
								detected_cube->scale(0),
								detected_cube->scale(1),
								detected_cube->scale(2);  // xyz roll pitch yaw scale

				// 转换到相机坐标系下的测量.
				cube_ground_value.fromMinimalVector(cube_pose);
				cube_local_meas = cube_ground_value.transform_to(curr_cam_pose_Twc); // measurement is in local camera frame

				// 如果对roll/pitch采样，则将其转换到正确的相机帧.
				if (detect_cuboid_obj.whether_sample_cam_roll_pitch)  //if camera roll/pitch is sampled, transform to the correct camera frame.
				{
					Vector3d new_camera_eulers =  detect_cuboid_obj.cam_pose_raw.euler_angle;
					new_camera_eulers(0) += detected_cube->camera_roll_delta; new_camera_eulers(1) += detected_cube->camera_pitch_delta;
					Matrix3d rotation_new = euler_zyx_to_rot<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
					Vector3d trans = transToWolrd.col(3).head<3>();
					g2o::SE3Quat curr_cam_pose_Twc_new(rotation_new,trans);
					cube_local_meas = cube_ground_value.transform_to(curr_cam_pose_Twc_new);
				}
				
				// NOTE 提案的误差，误差来自于  detected_cube 对象，其又来自于 frames_cuboids 序列，由前面的【立方体检测函数】得到
				proposal_error = detected_cube->normalized_error;
	      }
	  }
	  // 离线模式.
	  else
	  {
		  // @PARAM 离线已知的立方体位姿矩阵 offline_pred_frame_objects
		  // NOTE 读取 detect_cuboids_saved.txt 中的立方体的位姿.
	      int cube_obs_frame_id = offline_pred_frame_objects(offline_cube_obs_row_id,0);	// 读取第一列编号.
	      has_detected_cuboid = cube_obs_frame_id==frame_index;		// 每一帧都检测到了.
	      if (has_detected_cuboid)  // prepare object measurement   not all frame has observation!!
	      {
			    // 读取立方体的 9 自由度的位姿.
				VectorXd measure_data = offline_pred_frame_objects.row(offline_cube_obs_row_id);
				g2o::cuboid cube_ground_value; 
				Vector9d cube_pose;
				cube_pose<< measure_data(1),
							measure_data(2),
							measure_data(3),
							0,
							0,
							measure_data(4),
							measure_data(5),
							measure_data(6),
							measure_data(7);  // xyz roll pitch yaw scale
				
				// 将 9 自由度的位姿转换成 g2o::cuboid 中的形式.（世界坐标系）
				cube_ground_value.fromMinimalVector(cube_pose);

				// 读取每一帧相机的位姿.
				Eigen::VectorXd cam_pose_vec = init_frame_poses.row(frame_index);
				g2o::SE3Quat cam_val_Twc(cam_pose_vec.segment<7>(1)); // time x y z qx qy qz qw
				// NOTE 将立方体的位姿转换到相机坐标系
				cube_local_meas = cube_ground_value.transform_to(cam_val_Twc); // measurement is in local camera frame
				
				// NOTE 立方体提案的误差.
				proposal_error = measure_data(8);
				
				// 读取离线保存的带有提案的 2D 图像，并保存到 currframe->cuboids_2d_img 中.
				std::string detected_cube_2d_img_name = base_folder+"pred_3d_obj_overview/" + std::string(frame_index_c) + "_best_objects.jpg";
				currframe->cuboids_2d_img = cv::imread(detected_cube_2d_img_name, 1);
				
				offline_cube_obs_row_id++; // switch to next row  NOTE at most one object one frame in this data
	      }
	  }
	  
	  // STEP 4 立方体路标.
	  if (has_detected_cuboid)
	  {
		  // @PARAM 创建立方体路标.
	      object_landmark* localcuboid = new object_landmark();
	      
		  // 读取立方体的位姿，评估质量.
	      localcuboid->cube_meas = cube_local_meas;
	      localcuboid->meas_quality = (1-proposal_error+0.5)/2;  // initial error 0-1, higher worse,  now change to [0.5,1] higher, better
	    //   std::cout << "proposal_error:  " << proposal_error <<std::endl;
		//   std::cout << "meas_quality:  " << localcuboid->meas_quality <<std::endl;
		  currframe->observed_cuboids.push_back(localcuboid);
	  }
	  
	  // STEP 5 G2O 优化.
	  // STEP 5.1 第一帧时，设置 g2o 中的立方体顶点，本数据集中只有一个.
	  // set up g2o cube vertex. only one in this dataset
	  if (frame_index == 0)
	  {
		  // cuboid的初始位姿（世界坐标系）.
	      g2o::cuboid init_cuboid_global_pose = cube_local_meas.transform_from(curr_cam_pose_Twc);
	      vCube = new g2o::VertexCuboid();
	      vCube->setEstimate(init_cuboid_global_pose);
	      vCube->setId(0);
	      vCube->setFixed(false);
	      graph.addVertex(vCube);			// NOTE 添加顶点（立方体）
	  }
	  
	  // STEP 5.2 设置相机的顶点.
	  // set up g2o camera vertex
	  g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	  currframe->pose_vertex = vSE3;
	  vSE3->setId(frame_index+1);
	  graph.addVertex(vSE3);				// NOTE 添加顶点（相机）
	  // G2O 一般将顶点存储为从世界到相机的位姿.
	  vSE3->setEstimate(curr_cam_pose_Twc.inverse()); //g2o vertex usually stores world to camera pose.
	  vSE3->setFixed(frame_index==0);		
	  
	  // STEP 5.3 添加 G2O的边：相机-物体测量.
	  // add g2o camera-object measurement edges, if there is
	  if (currframe->observed_cuboids.size()>0)
	  {
			object_landmark* cube_landmark_meas = all_frames[frame_index]->observed_cuboids[0];

			g2o::EdgeSE3Cuboid* e = new g2o::EdgeSE3Cuboid();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vSE3 ));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vCube ));
			e->setMeasurement(cube_landmark_meas->cube_meas);
			e->setId(frame_index);

			// TODO inv_sigma 是啥？
			Vector9d inv_sigma;
			inv_sigma << 1,1,1,1,1,1,1,1,1;
			inv_sigma = inv_sigma * 2.0 * cube_landmark_meas->meas_quality;
			// 格式转换	Vector9d ——> Matrix9d
			Matrix9d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
			e->setInformation(info);
			graph.addEdge(e);				// NOTE 添加边
	  }
	  
	  // STEP 5.4 相机——顶点，相机-相机帧间旋转——边
	  // camera vertex, add cam-cam odometry edges
	  if (frame_index>0)
	  {
		g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( all_frames[frame_index-1]->pose_vertex ));
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>( all_frames[frame_index]->pose_vertex ));
		e->setMeasurement(odom_val);

		e->setId(total_frame_number+frame_index);
		Vector6d inv_sigma;
		inv_sigma << 1,1,1,1,1,1;
		inv_sigma = inv_sigma * 1.0;
		Matrix6d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
		e->setInformation(info);
		graph.addEdge(e);
	  }
	  // STEP 5.4 开始优化！！
	  graph.initializeOptimization();
	  graph.optimize(5); // do optimization!

	  // 优化之后相机的位姿，存储到当前帧序列 all_frames 中
	  // retrieve the optimization result, for debug visualization
	  for (int j=0;j<=frame_index;j++)
	  {
		  	// 优化之后的位姿（世界——相机）.
			all_frames[j]->cam_pose_Tcw = all_frames[j]->pose_vertex->estimate();
			// 相机——世界.
			all_frames[j]->cam_pose_Twc = all_frames[j]->cam_pose_Tcw.inverse();
	  }

	  // NOTE 优化之后的物体位姿 vCube->estimate() —— cube_pose_opti_history
	  object_landmark* current_landmark = new object_landmark();  
	  current_landmark->cube_vertex = new g2o::VertexCuboid();
	  current_landmark->cube_vertex->setEstimate(vCube->estimate());
	  cube_pose_opti_history[frame_index] = current_landmark;
	  
	  // NOTE 原始检测到的物体位姿 global_cube —— cube_pose_raw_detected_history
	  if (all_frames[frame_index]->observed_cuboids.size()>0)
	  {
		  // @PARAM cube_landmark_meas-当前帧的立方体路标的测量
	      object_landmark* cube_landmark_meas = all_frames[frame_index]->observed_cuboids[0];
	      g2o::cuboid local_cube = cube_landmark_meas->cube_meas;
	      g2o::cuboid global_cube = local_cube.transform_from(all_frames[frame_index]->cam_pose_Twc);

	      object_landmark* tempcuboids2 = new object_landmark();	  
		  tempcuboids2->cube_vertex = new g2o::VertexCuboid();
	      tempcuboids2->cube_vertex->setEstimate(global_cube);
	      cube_pose_raw_detected_history[frame_index] = tempcuboids2;
	  }
	  else
	      cube_pose_raw_detected_history[frame_index] = nullptr;
    }
    
	// STEP 6. 结束优化，开始可视化.
    cout<<"+++++++++++++Finish all optimization! Begin visualization.++++++++++++"<<endl;
    publish_all_poses(all_frames, cube_pose_opti_history,cube_pose_raw_detected_history,truth_frame_poses);      
}

// BRIEF
int main(int argc, char* argv[])
{
	// STEP 【1 ：初始化参数.】
    // 初始化节点与命名空间.
    ros::init(argc, argv, "object_slam");
    ros::NodeHandle nh;
    /** 初始化参数. 格式：nh.param ("name", name, value);
     *  @param	base_folder		【data文件目录】：~/catkin_object/src/cube_slam/object_slam/data
     *  @param	online_detect_mode	【检测模式】，默认为true，【在线检测】，false为使用离线相机位姿
     *  @param	save_results_to_txt	【是否保存结果】，默认为false
     */
    nh.param ("base_folder", base_folder, ros::package::getPath("object_slam")+"/data/");
    nh.param ("online_detect_mode", online_detect_mode, true);
    nh.param ("save_results_to_txt", save_results_to_txt, true);
    // 输出参数信息.
    cout<<""<<endl;
    cout<<"base_folder   "<<base_folder<<endl;
    if (online_detect_mode)
		ROS_WARN_STREAM("Online detect object mode !!\n");
    else
		ROS_WARN_STREAM("Offline read object mode !!\n");
	if (save_results_to_txt)
		ROS_WARN_STREAM("save results to output_cam_poses.txt !!\n");
    else
		ROS_WARN_STREAM("don`t save results to txt !!\n");
    
	// STEP 【2：读取参数.】
	// NOTE 在线模式下，不使用 pred_objs_txt 和 init_camera_pose ，仅使用 truth_camera_pose 中的第一帧
    /** 读取参数文件
     *  @param	pred_objs_txt		local ground frame中的立方体位姿.
     *  @param	init_camera_pose	离线相机位姿 (x y yaw=0, truth roll/pitch/height).
     *  @param	truth_camera_pose	真实的相机位姿. TODO 这两个相机位姿的区别.
     */
    std::string pred_objs_txt = base_folder+"detect_cuboids_saved.txt";   // saved cuboids in local ground frame.
    std::string init_camera_pose = base_folder+"pop_cam_poses_saved.txt"; // offline camera pose for cuboids detection (x y yaw=0, truth roll/pitch/height)
    std::string truth_camera_pose = base_folder+"truth_cam_poses.txt";
    
	/** 从各个文件中读取位姿矩阵.
     *  @param	pred_frame_objects	读取的已知的立方体位姿.
     *  @param	init_frame_poses	读取的已知的相机位姿.
     *  @param	truth_frame_poses	相机真实的位姿（仅使用第一帧的信息） NOTE .
     */
    Eigen::MatrixXd pred_frame_objects(100,10);  // 100 is some large row number, each row in txt has 10 numbers
    Eigen::MatrixXd init_frame_poses(100,8);	 // NOTE 需要给定列数，行数会自动读取！！
    Eigen::MatrixXd truth_frame_poses(100,8);
    if (!read_all_number_txt(pred_objs_txt,pred_frame_objects))
		return -1;
    if (!read_all_number_txt(init_camera_pose,init_frame_poses))
		return -1;
    if (!read_all_number_txt(truth_camera_pose,truth_frame_poses))
		return -1;
	//std::cout << truth_frame_poses << std::endl;
    // 输出三个文件的行数：read data size:  51  58  58.
    std::cout<<"read data size:  "<<pred_frame_objects.rows()<<"  "<<init_frame_poses.rows()<<"  "<<truth_frame_poses.rows()<<std::endl;
    
    // STEP 【3：传入参数开始增量式图优化.】
	std::cout << "+++++++++++++ 参数读取完毕，开始优化！！ ++++++++++++" << std::endl;
    incremental_build_graph(pred_frame_objects,init_frame_poses,truth_frame_poses);    
    
    return 0;
}
