#include "detect_3d_cuboid/object_3d_util.h"
#include "detect_3d_cuboid/matrix_utils.h"

#include <iostream>
// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace Eigen;
using namespace std;

// BRIEF 相似变换.
Matrix4d similarityTransformation(const cuboid& cube_obj)
{
    // 旋转.
    Matrix3d rot;
    rot <<  cos(cube_obj.rotY), -sin(cube_obj.rotY),     0, 
            sin(cube_obj.rotY),  cos(cube_obj.rotY),     0,
	                         0,                   0,     1;

    /* 将物体的尺寸设置为一个对角矩阵
        a   0   0
        0   b   0
        0   0   c  */
    Matrix3d scale_mat = cube_obj.scale.asDiagonal();

    // @PARAM 4*4的单位矩阵
    Matrix4d res = Matrix4d::Identity();
    res.topLeftCorner<3,3>() = rot * scale_mat;
    // 最后一列是物体的位置.
    res.col(3).head(3) = cube_obj.pos;

    // std::cout << "相似变换：\n" << res << std::endl;
    /*
    -0.151217    0.104412          0   -1.58339
    -0.0372463  -0.423907          0   0.373187
            0          0    0.300602   0.300602
            0          0           0          1
    */

    return res;
}

// BRIEF 输出立方体提案的信息.
void cuboid::print_cuboid()
{
    std::cout<<"printing cuboids info...."<<std::endl;
    std::cout<<"pos   "<<pos.transpose()<<std::endl;
    std::cout<<"scale   "<<scale.transpose()<<std::endl;
    std::cout<<"rotY   "<<rotY<<std::endl;
    std::cout<<"box_config_type   "<<box_config_type.transpose()<<std::endl;
    std::cout<<"box_corners_2d \n"<<box_corners_2d<<std::endl;
    std::cout<<"box_corners_3d_world \n"<<box_corners_3d_world<<std::endl;
}

// BRIEF  计算立方体的 3D 坐标.
Matrix3Xd compute3D_BoxCorner(const cuboid& cube_obj)
{
    //@PARAM    corners_body    存储立方体的3D坐标.
    MatrixXd corners_body;
    corners_body.resize(3,8);
    // 八个3D点       1   2   3   4   5   6   7   8
    corners_body <<  1,  1, -1, -1,  1,  1, -1, -1,
		             1, -1, -1,  1,  1, -1, -1,  1,
		            -1, -1, -1, -1,  1,  1,  1,  1;

    // 计算 3D 坐标                                         相似变换
    MatrixXd corners_world = homo_to_real_coord<double>(similarityTransformation(cube_obj) * real_to_homo_coord<double>(corners_body));
    
    return corners_world;
}

// BRIEF 输出n*2的边缘
// Output: n*2  each row is a edge's start and end pt id. 
// box_config_type  [configuration_id, vp_1_on_left_or_right]      cuboid struct has this field.
void get_object_edge_visibility( MatrixXi& visible_hidden_edge_pts,
                                 const Vector2d& box_config_type, 
                                 bool final_universal_object)
{   // 12 条边.
    visible_hidden_edge_pts.resize(12,2);

    if (final_universal_object) // final saved cuboid struct
    {  
        // 观察模式（3个面）
        if (box_config_type(0) == 1)  // look at get_cuboid_face_ids to know the faces and pt id using my old box format
        {
            // 如果vp1在左边.
            if (box_config_type(1)==1)
                //                          y    x    z    |  y    x    z  |           x  | y    x    y
                visible_hidden_edge_pts << 3,4, 4,1, 4,8,    1,2, 2,3, 2,6, 1,5, 3,7, 5,6, 6,7, 7,8, 8,5;   // TODO 1,5 3,7 ——> 3,5 1,7
            else
                visible_hidden_edge_pts << 2,3, 3,4, 3,7,    1,2, 1,4, 2,6, 1,5, 4,8, 5,6, 6,7, 7,8, 8,5;
	    }
        else
            visible_hidden_edge_pts << 2,3, 3,4, 4,1, 3,7, 4,8,    1,2, 2,6, 1,5, 5,6, 6,7, 7,8, 8,5;      
    }
    else// 2D box corners index only used in cuboids genetation process
    {  
        if (box_config_type(0)==1)
            visible_hidden_edge_pts<<7,8, 7,6, 7,1,    1,2, 2,3, 3,4, 4,1, 2,6, 3,5, 4,8, 5,8, 5,6; // hidden + visible
        else
            visible_hidden_edge_pts<<7,8, 7,6, 7,1, 8,4, 8,5,    1,2, 2,3, 3,4, 4,1, 2,6, 3,5, 5,6;
    }
    // std::cout << "visible_hidden_edge_pts：\n" << visible_hidden_edge_pts << std::endl;
}

// BRIEF  生成边缘信息.
// 输出：边缘边界 edge_markers ，每行：边缘起点，终点，类型.
// output: edge_markers  each row [ edge_start_pt_id, edge_end_pt_id,  edge_marker_type_id in line_marker_type ]
// box_config_type  [configuration_id, vp_1_on_left_or_right]      cuboid struct has this field.
void get_cuboid_draw_edge_markers(  MatrixXi& edge_markers,             /* 输出的边缘标记（xyz和是否可见） */
                                    const Vector2d& box_config_type,    /* 模式1 2，vp1的位置 */
                                    bool final_universal_object)        /* 是否是最终保存的提案 */
{
    // @PARAM  visible_hidden_edge_pts  线的两个端点
    MatrixXi visible_hidden_edge_pts;
    get_object_edge_visibility(visible_hidden_edge_pts, box_config_type, final_universal_object);

    // @PARAM  edge_line_markers  边的信息：颜色，是否可见.
    VectorXi edge_line_markers(12);

    // 最终确定的提案.
    if (final_universal_object)  // final saved cuboid struct  
    {
        // 情形 1 ，vp1在左边.
        if (box_config_type(0)==1)
        {
            if (box_config_type(1)==1)
                edge_line_markers << 4,2,6,3,1,5,5,5,3,1,3,1;		
            else
                edge_line_markers << 2,4,6,3,1,5,5,5,3,1,3,1;
        }
        else
            edge_line_markers << 2,4,2,6,6,3,5,5,3,1,3,1;
        //cout << "edge_line_markers:\n" << edge_line_markers << std::endl;
    }
    // 生成过程中的临时提案.
    else  // 2D box corners index only used in cuboids genetation process
    {
        if (box_config_type(0)==1)
            edge_line_markers<<4,2,6,1,3,1,3,5,5,5,1,3;   // each row: edge_start_id,edge_end_id,edge_marker_type_id
        else
            edge_line_markers<<4,2,6,6,2,1,3,1,3,5,5,3;
    }
    
    edge_markers.resize(12,3);
    edge_markers << visible_hidden_edge_pts, edge_line_markers;
    // cout << "edge_markers:\n" << edge_markers << std::endl;
    edge_markers.array() -=1;  // to match c++ index，每个元素-1
    /*
    3 4 4
    4 1 2
    4 8 6
    1 2 3
    2 3 1
    2 6 5
    1 5 5       // 3 5
    3 7 5       // 1 7
    5 6 3
    6 7 1
    7 8 3
    8 5 1
    */
}

// BRIEF 12条边绘制.
// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_cuboid_edges(  cv::Mat& plot_img, 
                                    const MatrixXi& box_corners_2d,     /* 2D 坐标 */
                                    const MatrixXi& edge_markers)       /* 边：第一列x,第二列y,第三列 类型（xyz轴是否可见）1-6*/
{
    // @PARAM   line_markers    存储矩形边的颜色和类型
    MatrixXi line_markers(6,4); // each row is  BGR, line_thickness线段粗细
    line_markers << 0,0,255,3,      // 1：红色，y轴，可见
                    0,0,255,0.3,    // 2：红色，y轴，不可见     粗细为 0.3（在背面看不到）
                    0,255,0,3,      // 3：绿色，x轴(长)，可见
                    0,255,0,0.3,    // 4：绿色，x轴(长)，不可见
                    255,0,0,3,      // 5：蓝色，z轴，可见
                    255,0,0,0.3;    // 6：蓝色，z轴，不可见

    // 绘制每一条边.
    for (int edge_id = 0; edge_id < edge_markers.rows(); edge_id++)
    {
        VectorXi edge_conds = edge_markers.row(edge_id);
        cv::line(   plot_img, 
                    cv::Point(box_corners_2d(0, edge_conds(0)), box_corners_2d(1, edge_conds(0))),  /* 第 edge_conds(0) 个点的xy坐标 */
                    cv::Point(box_corners_2d(0, edge_conds(1)), box_corners_2d(1, edge_conds(1))),  /* 第 edge_conds(1) 个点的xy坐标 */
                    // 颜色
                    cv::Scalar(line_markers(edge_conds(2),0), line_markers(edge_conds(2),1), line_markers(edge_conds(2),2)),
                    // 粗细（是否可见）
                    line_markers(edge_conds(2),3), 
                    CV_AA, //CV_AA, 
                    0);
        
        // 标注点
        // cv::putText(plot_img, 
        //             to_string(edge_conds(0) + 1), 
        //             cv::Point(box_corners_2d(0, edge_conds(0)), box_corners_2d(1, edge_conds(0))),
        //             2,      // fontFace
        //             0.8,    // fontScale
        //             cv::Scalar(255, 0, 0), 
        //             1);     // 粗细
        // cv::putText(plot_img, 
        //             to_string(edge_conds(1) + 1), 
        //             cv::Point(box_corners_2d(0, edge_conds(1)), box_corners_2d(1, edge_conds(1))),
        //             2,      // fontFace
        //             0.8,    // fontScale
        //             cv::Scalar(255, 0, 0), 
        //             1);     // 粗细
    }
}

// BRIEF 将立方体提案绘制在原图上.
void plot_image_with_cuboid(cv::Mat& plot_img, const cuboid* cube_obj)
{
    MatrixXi edge_markers;  
    get_cuboid_draw_edge_markers(edge_markers, cube_obj->box_config_type, true);
    plot_image_with_cuboid_edges(plot_img, cube_obj->box_corners_2d, edge_markers);
}

// BRIEF plot_image_with_edges() 函数在原图上绘制检测到的线段.
// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_edges(const cv::Mat& rgb_img, cv::Mat& output_img, MatrixXd& all_lines, const cv::Scalar& color)
{
    output_img = rgb_img.clone();
    for (int i = 0; i < all_lines.rows(); i++)
        cv::line(   output_img,
                    cv::Point(all_lines(i,0),all_lines(i,1)),
                    cv::Point(all_lines(i,2),all_lines(i,3)), 
                    color, 
                    2, 
                    16, 
                    0);
}

// BRIEF check_inside_box() 函数判断点 pt 是否在 box_left_top 和 box_right_bottom 组成的边框内.
bool check_inside_box(const Vector2d& pt, const Vector2d& box_left_top, const Vector2d& box_right_bottom)
{
    return box_left_top(0)<=pt(0) && pt(0)<=box_right_bottom(0) && box_left_top(1)<=pt(1) && pt(1)<=box_right_bottom(1);
}

// make sure edges start from left to right
// BRIEF align_left_right_edges()函数 确保存储的边缘两个端点是从左到右.
void align_left_right_edges(MatrixXd& all_lines)
{
    for (int line_id=0; line_id < all_lines.rows(); line_id++)
    {
        // 0 1， 2 3 要求第2个点的x坐标（2）要大于第一个点的 x 坐标（0）.
        if (all_lines(line_id,2) < all_lines(line_id,0))
        {
            Vector2d temp = all_lines.row(line_id).tail<2>();
            all_lines.row(line_id).tail<2>() = all_lines.row(line_id).head<2>();
            all_lines.row(line_id).head<2>() = temp;
        }
    }
}


void normalize_to_pi_vec(const VectorXd& raw_angles, VectorXd& new_angles)
{
    new_angles.resize(raw_angles.rows());
    for (int i=0;i<raw_angles.rows();i++)
	new_angles(i)=normalize_to_pi<double>(raw_angles(i));
}

// BRIEF 根据线段的 水平和竖直长度x_vec y_vec 计算出角度 all_angles
void atan2_vector(const VectorXd& y_vec, const VectorXd& x_vec, VectorXd& all_angles)
{
    all_angles.resize(y_vec.rows());
    for (int i=0;i<y_vec.rows();i++)
	    all_angles(i)=std::atan2(y_vec(i),x_vec(i));  // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
}

// remove the jumping angles from -pi to pi.   to make the raw angles smoothly change.
// BRIEF 从 -pi 到 pi 的顺序移除 jumping angles，使原始角度平滑变化.
void smooth_jump_angles(const VectorXd& raw_angles,VectorXd& new_angles)
{
    // 
    new_angles = raw_angles;
    if (raw_angles.rows()==0)
        return;

    double angle_base = raw_angles(0);  // choose a new base angle.   (assume that the all the angles lie in [-pi pi] around the base)
    std::cout << "angle_base: " << angle_base << std::endl;

    for (int i = 0; i < raw_angles.rows(); i++)
    {
        std::cout << "角度：\n" << (raw_angles(i) * 180)/M_PI << std::endl;
        if ( (raw_angles(i)-angle_base) < - M_PI )
            new_angles(i) = raw_angles(i) + 2 * M_PI;
        else if ( (raw_angles(i) - angle_base) > M_PI )
            new_angles(i) = raw_angles(i) - 2 * M_PI;
        // std::cout << raw_angles(i) << " - " << angle_base << " = " << raw_angles(i)-angle_base << std::endl;
        // 满足条件的边数：3
        // angle_base: 2.81415
        // 2.81415 - 2.81415 = 0
        // 2.5341 - 2.81415 = -0.280048
        // 2.6773 - 2.81415 = -0.136855
    }
}

// line_1  4d  line_segment2 4d  the output is float point.
// compute the intersection of line_1 (from start to end) with line segments (not infinite line). if not found, return [-1 -1]
// the second line segments are either horizontal or vertical.   a simplified version of lineSegmentIntersect

// BRIEF    seg_hit_boundary    检查消失点-上边缘采样点的射线是否与边界框的左右边界有交集，没有交集则返回 [-1 -1]. 需要判断线段是水平还是垂直线.
Vector2d seg_hit_boundary(const Vector2d& pt_start, const Vector2d& pt_end, const Vector4d& line_segment2 )
{
    // 线段 line_segment2 的起点和终点的y坐标.
    Vector2d boundary_bgn = line_segment2.head<2>();
    Vector2d boundary_end = line_segment2.tail<2>();

    // 消失点与上边缘采样点构成线段的长度（x和y的长度）.
    Vector2d direc = pt_end - pt_start;
    Vector2d hit_pt(-1,-1);
    
    // line equation is (p_u,p_v)+lambda*(delta_u,delta_v)  parameterized by lambda

    // 如果是水平边缘，两个点的 y 坐标相等.
    if ( boundary_bgn(1) == boundary_end(1) )   // if an horizontal edge
    {
        // 
        double lambd = (boundary_bgn(1)-pt_start(1))/direc(1);
        if (lambd >= 0)  // along ray direction
        {
            // @PARAM   hit_pt_tmp
            Vector2d hit_pt_tmp = pt_start + lambd * direc;
            if ( (boundary_bgn(0) <= hit_pt_tmp(0)) && (hit_pt_tmp(0) <= boundary_end(0)) )  // inside the segments
            {
                hit_pt = hit_pt_tmp;
                hit_pt(1) = boundary_bgn(1);  // floor operations might have un-expected things
            }
	    }
    }

    // 如果是垂直边缘.
    if ( boundary_bgn(0) == boundary_end(0) )   // if an vertical edge
    {
        double lambd=(boundary_bgn(0)-pt_start(0))/direc(0);
        if (lambd>=0)  // along ray direction
        {
            Vector2d hit_pt_tmp = pt_start+lambd*direc;
            if ( (boundary_bgn(1)<=hit_pt_tmp(1)) && (hit_pt_tmp(1)<=boundary_end(1)) )  // inside the segments
            {
                hit_pt = hit_pt_tmp;
                hit_pt(0)= boundary_bgn(0);  // floor operations might have un-expected things
            }
        }
    }
    return hit_pt;
}

// compute two line intersection points, a simplified version compared to matlab
// BRIEF    lineSegmentIntersect()    计算两条线的交点.
Vector2d lineSegmentIntersect(  const Vector2d& pt1_start, const Vector2d& pt1_end,     /* 线段 1*/
                                const Vector2d& pt2_start, const Vector2d& pt2_end,     /* 线段 2*/
			                    bool infinite_line)
{
    // treat as [x1 y1 x2 y2]    [x3 y3 x4 y4]
    double X2_X1 = pt1_end(0)-pt1_start(0);
    double Y2_Y1 = pt1_end(1)-pt1_start(1);

    double X4_X3 = pt2_end(0)-pt2_start(0);
    double Y4_Y3 = pt2_end(1)-pt2_start(1);

    double X1_X3 = pt1_start(0)-pt2_start(0);
    double Y1_Y3 = pt1_start(1)-pt2_start(1);

    double u_a = (X4_X3*Y1_Y3 - Y4_Y3*X1_X3) / (Y4_Y3*X2_X1 - X4_X3*Y2_Y1);
    double u_b = (X2_X1*Y1_Y3 - Y2_Y1*X1_X3) / (Y4_Y3*X2_X1 - X4_X3*Y2_Y1);      

    double INT_X = pt1_start(0) + X2_X1*u_a;
    double INT_Y = pt1_start(1) + Y2_Y1*u_a;
    double INT_B = double((u_a >= 0) && (u_a <= 1) && (u_b >= 0) && (u_b <= 1));

    if (infinite_line)
	  INT_B=1;
    
    return Vector2d(INT_X*INT_B, INT_Y*INT_B);      
}

Vector2f lineSegmentIntersect_f(const Vector2f& pt1_start, const Vector2f& pt1_end, const Vector2f& pt2_start, const Vector2f& pt2_end,
			      float& extcond_1, float& extcond_2, bool infinite_line)
{
    // treat as [x1 y1 x2 y2]    [x3 y3 x4 y4]
      float X2_X1 = pt1_end(0)-pt1_start(0);
      float Y2_Y1 = pt1_end(1)-pt1_start(1);
      float X4_X3 = pt2_end(0)-pt2_start(0);
      float Y4_Y3 = pt2_end(1)-pt2_start(1);
      float X1_X3 = pt1_start(0)-pt2_start(0);
      float Y1_Y3 = pt1_start(1)-pt2_start(1);
      float u_a = (X4_X3*Y1_Y3-Y4_Y3*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);
      float u_b = (X2_X1*Y1_Y3-Y2_Y1*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);      
      float INT_X = pt1_start(0)+X2_X1*u_a;
      float INT_Y = pt1_start(1)+Y2_Y1*u_a;
      float INT_B = float((u_a >= 0) && (u_a <= 1) && (u_b >= 0) && (u_b <= 1));
      if (infinite_line)
	  INT_B=1;
      
      extcond_1 = u_a; extcond_2 = u_b;
      return Vector2f(INT_X*INT_B, INT_Y*INT_B);      
}

cv::Point2f lineSegmentIntersect_f(const cv::Point2f& pt1_start, const cv::Point2f& pt1_end, const cv::Point2f& pt2_start, const cv::Point2f& pt2_end, 
			      float& extcond_1, float& extcond_2, bool infinite_line)
{
    // treat as [x1 y1 x2 y2]    [x3 y3 x4 y4]
      float X2_X1 = pt1_end.x-pt1_start.x;
      float Y2_Y1 = pt1_end.y-pt1_start.y;
      float X4_X3 = pt2_end.x-pt2_start.x;
      float Y4_Y3 = pt2_end.y-pt2_start.y;
      float X1_X3 = pt1_start.x-pt2_start.x;
      float Y1_Y3 = pt1_start.y-pt2_start.y;
      float u_a = (X4_X3*Y1_Y3-Y4_Y3*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);
      float u_b = (X2_X1*Y1_Y3-Y2_Y1*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);      
      float INT_X = pt1_start.x+X2_X1*u_a;
      float INT_Y = pt1_start.y+Y2_Y1*u_a;
      float INT_B = float((u_a >= 0) && (u_a <= 1) && (u_b >= 0) && (u_b <= 1));
      if (infinite_line)
	  INT_B=1;
      
      extcond_1 = u_a; extcond_2 = u_b;
      return cv::Point2f(INT_X*INT_B, INT_Y*INT_B);

}

// BRIEF merge_break_lines() 函数将短边合并成长边.
// merge short edges into long. edges n*4  each edge should start from left to right! 
void merge_break_lines( const MatrixXd& all_lines,          /*输入的所有在矩阵框内的线段矩阵*/
                        MatrixXd& merge_lines_out,          /*输出的合并后的线段矩阵*/
                        double pre_merge_dist_thre,         /*两条线段之间的距离阈值 20 像素*/
		                double pre_merge_angle_thre_degree, /*角度阈值 5°*/
                        double edge_length_threshold)       /*长度阈值 30*/
{
    bool can_force_merge = true;
    merge_lines_out = all_lines;
    // 线段条数：total_line_number 将越来越小，merge_lines_out 不变.
    int total_line_number = merge_lines_out.rows();  // line_number will become smaller and smaller, merge_lines_out doesn't change
    int counter = 0;
    // 角度阈值，转换成弧度.
    double pre_merge_angle_thre = pre_merge_angle_thre_degree/180.0*M_PI;

    // STEP 【1.线段融合.】
    while ((can_force_merge) && (counter<500))
    {
	    counter++;
	    can_force_merge=false;
        // 线段向量：所有线段的右边点的坐标 - 左边点的坐标 = 每条线段的水平x长度和竖直y长度.
	    MatrixXd line_vector = merge_lines_out.topRightCorner(total_line_number,2) - merge_lines_out.topLeftCorner(total_line_number,2);

        //  @PARAM all_angles 计算【每条线段的角度】. 
	    VectorXd all_angles; 
        // 根据线段的 x 和 y 长度计算出角度（°）.
        atan2_vector(line_vector.col(1),line_vector.col(0),all_angles); // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
	    
        // 处理每一条线段.
        for (int seg1 = 0;seg1 < total_line_number - 1; seg1++) 
        {
		    for (int seg2 = seg1+1; seg2 < total_line_number; seg2++)
            {
                // 相邻两条选段的角度差 angle_diff.
                double diff = std::abs(all_angles(seg1) - all_angles(seg2));
                double angle_diff = std::min(diff, M_PI - diff);

                // STEP 【1.1 先判断角度偏差】如果两条线段的角度误差小于 5°.
                if (angle_diff < pre_merge_angle_thre)
                {
                    // dist_1ed_to_2：线1尾到线2头的距离；
                    // dist_2ed_to_1：线2尾到线1头的距离；
                    double dist_1ed_to_2 = (merge_lines_out.row(seg1).tail(2) - merge_lines_out.row(seg2).head(2)).norm();
                    double dist_2ed_to_1 = (merge_lines_out.row(seg2).tail(2) - merge_lines_out.row(seg1).head(2)).norm();
                    
                    // STEP 【1.2 再判断距离偏差】如果两条线段之间的距离阈值小于距离阈值 pre_merge_dist_thre 20 像素
                    if ((dist_1ed_to_2 < pre_merge_dist_thre) || (dist_2ed_to_1 < pre_merge_dist_thre))
                    {
                        // 确定融合之后的线段的两个端点.
                        Vector2d merge_start, merge_end;
                        if (merge_lines_out(seg1,0) < merge_lines_out(seg2,0))
                            merge_start = merge_lines_out.row(seg1).head(2);
                        else
                            merge_start = merge_lines_out.row(seg2).head(2);
                        if (merge_lines_out(seg1,2) > merge_lines_out(seg2,2))
                            merge_end = merge_lines_out.row(seg1).tail(2);
                        else
                            merge_end = merge_lines_out.row(seg2).tail(2);
                        
                        // 融合之后的新的线段的角度 merged_angle.
                        double merged_angle = std::atan2(merge_end(1)-merge_start(1),merge_end(0)-merge_start(0));
                        
                        // 计算线段 1 与合并之后的线段的角度偏差 merge_angle_diff.
                        double temp = std::abs(all_angles(seg1) - merged_angle);
                        double merge_angle_diff = std::min( temp, M_PI-temp );
                        
                        // NOTE 将融合的线段存储在 merge_lines_out 中，并减小 merge_lines_out 的大小（剔除了一条）.
                        if (merge_angle_diff < pre_merge_angle_thre)
                        {
                            merge_lines_out.row(seg1).head(2) = merge_start;
                            merge_lines_out.row(seg1).tail(2) = merge_end;
                            fast_RemoveRow(merge_lines_out, seg2, total_line_number);  //also decrease  total_line_number
                            can_force_merge = true;
                            break;
                        }
                    }
                }
		    }   // 循环 2
            if (can_force_merge)
                break;			
	    }   // 循环 1
    }// NOTE 线段融合 END.
    // 使用 LSD 的时候短线段比较多，合并比较明显，使用 edline 检测到的线段比较完整所以效果不是很明显.
    std::cout<<"合并之前的线段数：" << all_lines.rows() + 1 << std::endl;
    std::cout<<"合并之后的线段数：" << total_line_number + 1 << std::endl;

    // STEP 【2.长度筛选】
    if (edge_length_threshold > 0)
    {
        // 重新计算合并之后的线段向量：所有线段的右边点的坐标 - 左边点的坐标 = 每条线段的水平x长度和竖直y长度.
        MatrixXd line_vectors = merge_lines_out.topRightCorner(total_line_number,2) - merge_lines_out.topLeftCorner(total_line_number,2);
        // @PARAM line_lengths 存储每条线段的长度
        VectorXd line_lengths = line_vectors.rowwise().norm();
        // std::cout << "x y 投影长度 line_vectors:\n" << line_vectors << std::endl;
        // std::cout << "线段长度 line_lengths:\n" << line_lengths << std::endl;

        int long_line_number = 0;
        MatrixXd long_merge_lines(total_line_number, 4);
        for (int i = 0; i < total_line_number; i++)
        {
            // 如果线段长度大于阈值.
            if (line_lengths(i) > edge_length_threshold)
            {
                long_merge_lines.row(long_line_number) = merge_lines_out.row(i);
                long_line_number++;
            }
        }
        // 将长度满足要求的挑选出来.
        merge_lines_out = long_merge_lines.topRows(long_line_number);
    }
    else
	    merge_lines_out.conservativeResize(total_line_number,NoChange);
    std::cout<<"长度筛选之后的线段数：" << merge_lines_out.rows() + 1 << std::endl;
}

// VPs 3*2   edge_mid_pts: n*2   vp_support_angle_thres 1*2
// output: 3*2  each row is a VP's two boundary supported edges' angle.  if not found, nan for that entry
// BRIEF 3*2的矩阵，搜索可能构造该消失点的线段（满足角度差）
Eigen::MatrixXd VP_support_edge_infos(  Eigen::MatrixXd& VPs,                   /* 消失点矩阵 3*2 */
                                        Eigen::MatrixXd& edge_mid_pts,          /* 每条线段的中点 n×2 */
                                        Eigen::VectorXd& edge_angles,           /* 每条线段的偏角 n×1 */
				                        Eigen::Vector2d vp_support_angle_thres) /* 消失点与边的夹角阈值*/
{
    MatrixXd all_vp_bound_edge_angles = MatrixXd::Ones(3,2) * nan(""); // initialize as nan  use isnan to check
    if (edge_mid_pts.rows() > 0)
    {
        // 分别处理三个消失点.
        for (int vp_id = 0; vp_id < VPs.rows(); vp_id++)
        {
            // @PARAM   vp_angle_thre   夹角阈值.
            double vp_angle_thre;
            if (vp_id!=2)   /* 消失点 1 2 的夹角阈值 15.*/
                vp_angle_thre = vp_support_angle_thres(0)/180.0*M_PI;
            else            /* 消失点 3 的夹角阈值 10.*/
                vp_angle_thre = vp_support_angle_thres(1)/180.0*M_PI;
        
            std::vector<int> vp_inlier_edge_id;                             // 在范围内的边的 id.
            VectorXd vp_edge_midpt_angle_raw_inlier(edge_angles.rows());    // 边与第 vp_id 个消失点角度差在范围内的角度矩阵.

            // 消失点与每条边的夹角.
            for (int edge_id = 0; edge_id < edge_angles.rows(); edge_id++)
            {
                // @PARAM   vp1_edge_midpt_angle_raw_i   消失点到边的中点的角度. 
                double vp1_edge_midpt_angle_raw_i = atan2( edge_mid_pts(edge_id,1) - VPs(vp_id,1), edge_mid_pts(edge_id,0) - VPs(vp_id,0) );
                // std::cout << "vp1_edge_midpt_angle_raw_i:\n" << vp1_edge_midpt_angle_raw_i << std::endl;

                // @PARAM   vp1_edge_midpt_angle_norm_i  标准化之后的角度（-90 ~90）.
                double vp1_edge_midpt_angle_norm_i = normalize_to_pi<double>(vp1_edge_midpt_angle_raw_i);

                // @PARAM   angle_diff_i    消失点_中点的角度 与 线段的角度差.
                double angle_diff_i = std::abs(edge_angles(edge_id) - vp1_edge_midpt_angle_norm_i);
                angle_diff_i = std::min(angle_diff_i,M_PI-angle_diff_i);
                
                // NOTE 如果角度差小于阈值， 保存下与第 edge_id 条边中点的角度，也就是这条边可能是形成该消失点的边.
                if (angle_diff_i < vp_angle_thre)
                {
                    vp_edge_midpt_angle_raw_inlier(vp_inlier_edge_id.size()) = vp1_edge_midpt_angle_raw_i;
                    vp_inlier_edge_id.push_back(edge_id);
                }
            }

            // 如果存在在角度阈值内的线段.
            if (vp_inlier_edge_id.size() > 0) // if found inlier edges
            {
                // @PARAM   vp1_edge_midpt_angle_raw_inlier_shift   平滑处理之后的角度.
                VectorXd vp1_edge_midpt_angle_raw_inlier_shift; 
                // TODO 角度平滑变化. 什么作用没太理解？？
                smooth_jump_angles( vp_edge_midpt_angle_raw_inlier.head(vp_inlier_edge_id.size()),
                                    vp1_edge_midpt_angle_raw_inlier_shift);
                std::cout << "满足条件的边的数量：" << vp1_edge_midpt_angle_raw_inlier_shift.size() << std::endl;

                // NOTE 如果有多条边满足要求（可能检测到多条支撑线），比较得到角度最大和最小的，作为两条支撑线.
                // 角度最大和最小的边的id
                int vp1_low_edge_id;	
                vp1_edge_midpt_angle_raw_inlier_shift.maxCoeff(&vp1_low_edge_id);
                int vp1_top_edge_id;	
                vp1_edge_midpt_angle_raw_inlier_shift.minCoeff(&vp1_top_edge_id);

                // TODO 第 2 3 个消失点时交换最大和最小值
                if (vp_id > 0)
                    std::swap(vp1_low_edge_id, vp1_top_edge_id);  // match matlab code
                
                // NOTE 输出：消失点两边的夹角.
                all_vp_bound_edge_angles(vp_id,0) = edge_angles(vp_inlier_edge_id[vp1_low_edge_id]);   // it will be 0*1 matrix if not found inlier edges.
                all_vp_bound_edge_angles(vp_id,1) = edge_angles(vp_inlier_edge_id[vp1_top_edge_id]);
            }
        }
    }
    return all_vp_bound_edge_angles;
}

// BRIEF 计算与距离变换图的距离.
double box_edge_sum_dists(  const cv::Mat& dist_map,            /* 距离变换图 */
                            const MatrixXd& box_corners_2d,     /* 8 个顶点的 2D坐标 */
                            const MatrixXi& edge_pt_ids,        /* 可见的边 */
                            bool  reweight_edge_distance)
{
    /*
    给定一些边，在线采样一些点，求和与 dist_map 的距离；
    输入：可见的边
    对于情形 1 ，相比情形 2 有更多的可见边缘，需要对边进行重新加权.
    give some edges, sample some points on line then sum up distance from dist_map
    input: visible_edge_pt_ids is n*2  each row stores an edge's two end point's index from box_corners_2d
    if weight_configs: for configuration 1, there are more visible edges compared to configuration2, so we need to re-weight
    [1 2;2 3;3 4;4 1;2 6;3 5;4 8;5 8;5 6];  reweight vertical edge id 5-7 by 2/3, horizontal edge id 8-9 by 1/2
    */

    float sum_dist = 0;
    // 遍历每一条可见边.
    for (int edge_id = 0; edge_id < edge_pt_ids.rows(); edge_id++)
    {
        // 每条可见边的两个点的 x 坐标的 y 坐标.
        Vector2d corner_tmp1 = box_corners_2d.col(edge_pt_ids(edge_id,0));
        Vector2d corner_tmp2 = box_corners_2d.col(edge_pt_ids(edge_id,1));

        for (double sample_ind = 0; sample_ind < 11; sample_ind++)
        {
            // 在线段上采样1个点   sample_pt.
            Vector2d sample_pt = sample_ind/10.0 * corner_tmp1 + (1-sample_ind/10.0) * corner_tmp2;

            // NOTE 计算距离.
            float dist1 = dist_map.at<float>(int(sample_pt(1)),int(sample_pt(0)));  //make sure dist_map is float type
            
            // 是否重新加权
            // TODO 第5,6,7条边的测量更值得信赖？？
            if (reweight_edge_distance)
            {
                if ((4<=edge_id) && (edge_id<=5))       // 对第 5,6 条边 × 1.5
                    dist1 = dist1 * 3.0 / 2.0;
                if (6==edge_id)                         // 对第 7 条边  × 2
                    dist1 = dist1 * 2.0;
            }

            sum_dist = sum_dist + dist1;
        }
    }
    return double(sum_dist);
}

// BRIEF 立方体边缘角度与消失点所对应的角度对齐误差.用于评估立方体质量.
double box_edge_alignment_angle_error(  const MatrixXd& all_vp_bound_edge_angles,   /* 消失点与边的两个角度 */
                                        const MatrixXi& vps_box_edge_pt_ids,        /* 每个消失点来源的两条边 */
                                        const MatrixXd& box_corners_2d)             /* 8 个顶点的 2D坐标 */
{
// compute the difference of box edge angle with angle of actually VP aligned image edges. for evaluating the box
// all_vp_bound_edge_angles: VP aligned actual image angles. 3*2  if not found, nan.      box_corners_2d: 2*8
// vps_box_edge_pt_ids: % six edges. each row represents two edges [e1_1 e1_2   e2_1 e2_2;...] of one VP
    double total_angle_diff = 0;
    double not_found_penalty = 30.0/180.0*M_PI*2;    // if not found any VP supported lines, give each box edge a constant cost (45 or 30 ? degree)
    
    // 遍历 3 个消失点分别对应的两条边（四个点）.
    for (int vp_id = 0; vp_id < vps_box_edge_pt_ids.rows(); vp_id++)
    {
        // 读取消失点对应的两个角度.
        Vector2d vp_bound_angles = all_vp_bound_edge_angles.row(vp_id);

        std::vector<double> vp_bound_angles_valid;
        
        // 有效角度.
        for (int i = 0; i < 2; i++)
            if (!std::isnan(vp_bound_angles(i)))
                vp_bound_angles_valid.push_back(vp_bound_angles(i));

        if (vp_bound_angles_valid.size() > 0) //  exist valid edges
        {
            // 分别得到 由 1 2 或 3 4 个点构成的两条线.
            for (int ee_id = 0; ee_id < 2; ee_id++) // find cloeset from two boundary edges. we could also do left-left right-right compare. but pay close attention different vp locations  
            {
                Vector2d two_box_corners_1 = box_corners_2d.col( vps_box_edge_pt_ids(vp_id, 2*ee_id) );     // 第 1（或3）个点[ x1;y1 ]
                Vector2d two_box_corners_2 = box_corners_2d.col( vps_box_edge_pt_ids(vp_id, 2*ee_id+1) );   // 第 2（或4）个点[ x2;y2 ]
                
                // @PARAM   box_edge_angle      边的角度.
                double box_edge_angle = normalize_to_pi(atan2(two_box_corners_2(1) - two_box_corners_1(1), 
                                                        two_box_corners_2(0) - two_box_corners_1(0)));  // [-pi/2 -pi/2]

                double angle_diff_temp = 100;
                for (int i = 0; i < vp_bound_angles_valid.size(); i++)
                {
                    // NOTE 【计算角度误差】，形成消失点的边的角度-消失点与检测到的边缘的角度.
                    double temp = std::abs(box_edge_angle - vp_bound_angles_valid[i]);

                    temp = std::min( temp, M_PI-temp );
                    if (temp<angle_diff_temp)
                        angle_diff_temp = temp;
                }
                    total_angle_diff = total_angle_diff + angle_diff_temp;
            }
        }
        // NOTE 如果没有找到形成消失点的边缘，则赋予固定的偏差.
        else
            total_angle_diff=total_angle_diff+not_found_penalty;
    }
    return total_angle_diff;
}

// BRIEF 加权不同的误差评估 weighted sum different score
void fuse_normalize_scores_v2(  const VectorXd& dist_error,         /* 距离误差 */           
                                const VectorXd& angle_error,        /* 角度误差 */
                                VectorXd& combined_scores,          /* 综合得分 */    
                                std::vector<int>& final_keep_inds,  /* 最终纳入计算的测量的ID */
			                    double weight_vp_angle,             /* 角度误差的权重 */
                                bool whether_normalize)             /* 是否归一化两个误差 */
{
    // 原始测量的次数
    int raw_data_size = dist_error.rows();

    if (raw_data_size > 4)
    {
        // @PARAM breaking_num  需要排序的数量：总数据量的 2/3.
        int breaking_num = round(float(raw_data_size)/3.0*2.0);

        // 从 0 开始生成一个有序向量：0,1，2 ... 4137 ...
        std::vector<int> dist_sorted_inds(raw_data_size); 
        std::iota(dist_sorted_inds.begin(), dist_sorted_inds.end(), 0);

        std::vector<int> angle_sorted_inds = dist_sorted_inds;
        
        // NOTE 对距离误差 dist_error 的前 breaking_num（2/3的量）递增排序.
        sort_indexes(   dist_error, 
                        dist_sorted_inds,   /* 排序依据 */
                        breaking_num);      /* 排序前breaking_num项 */     

        // NOTE 对角度误差 angle_error 的前 breaking_num（2/3的量）递增排序.
        sort_indexes(   angle_error, 
                        angle_sorted_inds, 
                        breaking_num);

        // 保存前 2/3 的数据.
        // @PARAM   dist_keep_inds  距离误差前2/3数据的ID（在dist_error中）
        std::vector<int> dist_keep_inds = std::vector<int>( dist_sorted_inds.begin(),
                                                            dist_sorted_inds.begin() + breaking_num-1);  // keep best 2/3

        // for(int i = 0; i <= dist_keep_inds.size(); i++)
        //     std::cout << "ID：\t" << dist_keep_inds[i] << "  " << "距离误差：\t" << dist_error(dist_keep_inds[i]) << std::endl;

        // 如果angle_error中第 angle_sorted_inds[breaking_num-1] 误差大于 angle_sorted_inds[breaking_num-2]
        if ( angle_error(angle_sorted_inds[breaking_num-1]) > angle_error(angle_sorted_inds[breaking_num-2]) )
        {
            // @PARAM    angle_keep_inds    角度误差前2/3数据的ID（在angle_error中）
            std::vector<int> angle_keep_inds = std::vector<int>(    angle_sorted_inds.begin(),
                                                                    angle_sorted_inds.begin() + breaking_num-1);  // keep best 2/3
            
            // 对距离和角度ID进行重新排序.
            std::sort(dist_keep_inds.begin(),dist_keep_inds.end());
            std::sort(angle_keep_inds.begin(),angle_keep_inds.end());

            // NOTE 寻找两个序列的交集：final_keep_inds.
            std::set_intersection(  dist_keep_inds.begin(), 
                                    dist_keep_inds.end(),
                                    angle_keep_inds.begin(), 
                                    angle_keep_inds.end(),
                                    std::back_inserter(final_keep_inds));
        }
        else  //don't need to consider angle.   my angle error has maximum. may already saturate at breaking pt.
        {
            final_keep_inds = dist_keep_inds;
        }
    }
    else
    {
	    final_keep_inds.resize(raw_data_size);   //don't change anything.
	    std::iota(final_keep_inds.begin(), final_keep_inds.end(), 0);
    }
    
    // 距离和角度误差都较小的 ID.
    int new_data_size = final_keep_inds.size();

    // find max/min of kept errors.
    double min_dist_error = 1e6; 
    double max_dist_error = -1;
    double min_angle_error = 1e6;
    double max_angle_error = -1;

    VectorXd dist_kept(new_data_size);  
    VectorXd angle_kept(new_data_size);

    // STEP 找到距离和角度误差最大最小值.
    for (int i = 0; i < new_data_size; i++)
    {
        double temp_dist = dist_error(final_keep_inds[i]);	
        double temp_angle = angle_error(final_keep_inds[i]);
        min_dist_error = std::min(min_dist_error,temp_dist);	
        max_dist_error = std::max(max_dist_error,temp_dist);
        min_angle_error = std::min(min_angle_error,temp_angle); 
        max_angle_error = std::max(max_angle_error,temp_angle);
        dist_kept(i) = temp_dist;  
        angle_kept(i) = temp_angle;
    }
    
    // STEP 误差归一化.
    if (whether_normalize && (new_data_size > 1))
    {
        // 距离误差             （所有的距离 - 最小距离值）/ (最大距离 - 最小距离)
        combined_scores  = (dist_kept.array() - min_dist_error) / (max_dist_error - min_dist_error);
        if ((max_angle_error - min_angle_error) > 0)
        {
            // 角度误差        （所有角度误差 - 最小角度误差） / （最大角度误差 - 最小角度误差）
            angle_kept = (angle_kept.array() - min_angle_error) / (max_angle_error - min_angle_error);

            // NOTE 联合评分，（距离误差 + 角度权重×角度误差）/（1 + 角度权重）
            combined_scores = (combined_scores + weight_vp_angle * angle_kept) / (1 + weight_vp_angle);
        }
        else
            combined_scores = (combined_scores + weight_vp_angle*angle_kept)/(1+weight_vp_angle);
    }
    else
	    combined_scores = (dist_kept + weight_vp_angle * angle_kept) / (1 + weight_vp_angle);    
}

// BRIEF 射线为 3×n ，每列都是从原点开始的一条射线  平面（4*1），计算射线的交点 3×n.
//rays is 3*n, each column is a ray staring from origin  plane is (4，1） parameters, compute intersection  output is 3*n 
void ray_plane_interact(const MatrixXd &rays,           
                        const Eigen::Vector4d &plane,
                        MatrixXd &intersections)
{
    VectorXd frac = -plane[3] / (plane.head(3).transpose() * rays).array();   //n*1 
    intersections = frac.transpose().replicate<3,1>().array() * rays.array();
}

// BRIEF 在 3D空间中计算与平面交线.
// compute ray intersection with plane in 3D.
// transToworld: 4*4 camera pose.   invK: inverse of calibration.   plane: 1*4  plane equation in sensor frame. 
// pixels  2*n; each column is a pt [x;y] x is horizontal,y is vertical   outputs: pts3d 3*n in world frame
void plane_hits_3d( const Matrix4d& transToWolrd,   /* 4*4 的相机位姿*/
                    const Matrix3d& invK,           /* 相机内参的逆矩阵 */
                    const Vector4d& plane_sensor,   /* 传感器坐标系中的 1*4 平面 */
                    MatrixXd pixels,                /* 像素 2×n ，每列表示一个点（x,y）*/
                    Matrix3Xd& pts_3d_world)        /* 输出：世界坐标系中的 3D 坐标点 */
{
    // 将之前的一列两行表示的点的坐标变成一列三行的形式.
    pixels.conservativeResize(3,NoChange);

    // pixels.cols() 列的数量
    // 将 pixels 的第三行设置为 pixels.cols() 个 1.
    pixels.row(2) = VectorXd::Ones(pixels.cols());

    MatrixXd pts_ray = invK * pixels;    //each column is a 3D world coordinate  3*n    	
    MatrixXd pts_3d_sensor;  
    ray_plane_interact(pts_ray, plane_sensor, pts_3d_sensor);

    pts_3d_world = homo_to_real_coord<double>(transToWolrd*real_to_homo_coord<double>(pts_3d_sensor)); //
}

// BRIEF 计算 wall_plane 的方程.
Vector4d get_wall_plane_equation(const Vector3d& gnd_seg_pt1, const Vector3d& gnd_seg_pt2)
// 1*6 a line segment in 3D. [x1 y1 z1  x2 y2 z2]  z1=z2=0  or  two 1*3
{
    // 线段与[0,0,1]的叉积？
    Vector3d partwall_normal_world = (gnd_seg_pt1 - gnd_seg_pt2).cross(Vector3d(0,0,1)); // [0,0,1] is world ground plane
    
    partwall_normal_world.array() /= partwall_normal_world.norm();

    double dist = -partwall_normal_world.transpose()*gnd_seg_pt1;
    
    Vector4d plane_equation;
    plane_equation << partwall_normal_world, dist;        // wall plane in world frame
    
    if (dist < 0)
        plane_equation = -plane_equation;   // make all the normal pointing inside the room. neamly, pointing to the camera
    return plane_equation;
}

// BRIEF    getVanishingPoints()    【消失点计算】.
void getVanishingPoints(const Matrix3d& KinvR,  /* Kalib*invR */
                        double yaw_esti,        /* 采样的物体偏航角 */
                        Vector2d& vp_1,         /* 输出的消失点 */
                        Vector2d& vp_2, 
                        Vector2d& vp_3)
{
    vp_1 = homo_to_real_coord_vec<double>( KinvR * Vector3d(cos(yaw_esti), sin(yaw_esti), 0) );     // for object x axis
    vp_2 = homo_to_real_coord_vec<double>( KinvR * Vector3d(-sin(yaw_esti), cos(yaw_esti), 0) );    // for object y axis
    vp_3 = homo_to_real_coord_vec<double>( KinvR * Vector3d(0,0,1) );                               // for object z axis
}

// box_corners_2d_float is 2*8    change to my object struct from 2D box corners.
// BRIEF    由2D顶点恢复出 3D 立方体信息.
void change_2d_corner_to_3d_object( const MatrixXd& box_corners_2d_float,   /* 8 个点的 2D 坐标*/
                                    const Vector3d& configs,                /* 模式，vp1的位置，偏航角*/
                                    const Vector4d& ground_plane_sensor,    /* 相机系下的地平面*/
				                    const Matrix4d& transToWolrd,           /* 相机旋转 */
                                    const Matrix3d& invK,                   /* 相机内参的逆矩阵 */
                                    Eigen::Matrix<double, 3, 4>& projectionMatrix,  /* 投影矩阵 */
				                    cuboid& sample_obj)                     /* 3D提案 */
{
    // @PARAM obj_gnd_pt_world_3d   计算世界坐标系中的 3D 点（立方体底部） .
    Matrix3Xd obj_gnd_pt_world_3d; 
    plane_hits_3d(  transToWolrd,                      /* 相机旋转矩阵 */
                    invK,                              /* 相机内参的逆矩阵 */
                    ground_plane_sensor,               /* 相机系下的地平面*/
                    box_corners_2d_float.rightCols(4), /* 立方体底部的 4 个 2D 点 */
                    obj_gnd_pt_world_3d);              /* 立方体底部的 4 个 3D 点 *///% 3*n each column is a 3D point  floating point
    
    // STEP 通过点 5-8 计算长度的一半
    double length_half = (obj_gnd_pt_world_3d.col(0)-obj_gnd_pt_world_3d.col(3)).norm()/2;  // along object x direction   corner 5-8
    // STEP 通过点 5-6 计算宽度的一半
    double width_half = (obj_gnd_pt_world_3d.col(0)-obj_gnd_pt_world_3d.col(1)).norm()/2;  // along object y direction   corner 5-6
    
    // 通过第 5,6 点计算世界坐标系和相机坐标系中的 wall_plane
    Vector4d partwall_plane_world = get_wall_plane_equation(obj_gnd_pt_world_3d.col(0),obj_gnd_pt_world_3d.col(1));//% to compute height, need to unproject-hit-planes formed by 5-6 corner
    Vector4d partwall_plane_sensor = transToWolrd.transpose()*partwall_plane_world;  // wall plane in sensor frame
    
    // @PARAM obj_top_pt_world_3d   计算世界坐标系中的 3D 点（立方体顶部） .
    Matrix3Xd obj_top_pt_world_3d; 
    plane_hits_3d(  transToWolrd,
                    invK,
                    partwall_plane_sensor,
                    box_corners_2d_float.col(1),
                    obj_top_pt_world_3d);  // should match obj_gnd_pt_world_3d  % compute corner 2
    
    // STEP 计算高度的一半，obj_top_pt_world_3d(2, 0)是立方体顶部第一个点的z坐标.
    double height_half = obj_top_pt_world_3d(2, 0)/2;
    
    // 顶部四个点的x和y坐标平均值.
    double mean_obj_x = obj_gnd_pt_world_3d.row(0).mean(); 
    double mean_obj_y = obj_gnd_pt_world_3d.row(1).mean();
    
    double vp_1_position = configs(1);  // 消失点 1 的位置.
    double yaw_esti = configs(2);       // 采样的偏航角.

    // STEP 物体的9自由度表示.
    // 物体的位置：x,y（平均值），z（高度的一半）
    sample_obj.pos = Vector3d(mean_obj_x, mean_obj_y, height_half);  
    // 方向.
    sample_obj.rotY = yaw_esti;
    // 尺度.
    sample_obj.scale = Vector3d(length_half,width_half,height_half);
    // 模式.
    sample_obj.box_config_type = configs.head<2>();

    // @PARAM   cuboid_to_raw_boxstructIds   八个点的编号. 
    VectorXd cuboid_to_raw_boxstructIds(8);
    if (vp_1_position==1)  // vp1 on left, for all configurations
        cuboid_to_raw_boxstructIds << 6, 5, 8, 7, 2, 3, 4, 1;
    if (vp_1_position==2)  // vp1 on right, for all configurations
        cuboid_to_raw_boxstructIds << 5, 6, 7, 8, 3, 2, 1, 4;

    // 将float类型的 2D坐标转换成 int 类型.
    Matrix2Xi box_corners_2d_int = box_corners_2d_float.cast<int>();
    sample_obj.box_corners_2d.resize(2,8);

    // 将8个点的 2D 坐标按照编号存储在对象中.
    for (int i = 0; i < 8; i++)
	    sample_obj.box_corners_2d.col(i) = box_corners_2d_int.col( cuboid_to_raw_boxstructIds(i)-1 ); // minius one to match index
    
    // NOTE 计算物体 8 个点的 3D 坐标.
    sample_obj.box_corners_3d_world = compute3D_BoxCorner(sample_obj);
}


float bboxOverlapratio(const cv::Rect& rect1, const cv::Rect& rect2)
{
    int overlap_area = (rect1&rect2).area();
    return (float)overlap_area/((float)(rect1.area()+rect2.area()-overlap_area));
}


int pointBoundaryDist(const cv::Rect& rect, const cv::Point2f& kp )
{
    int mid_x = rect.x + rect.width/2;
    int mid_y = rect.y + rect.height/2;
    int min_x_bound_dist = 0;int min_y_bound_dist = 0;
    if (kp.x<mid_x)
	min_x_bound_dist = abs(kp.x-rect.x);
    else
	min_x_bound_dist = abs(kp.x-rect.x-rect.width);
    if (kp.y<mid_y)
	min_y_bound_dist = abs(kp.y-rect.y);
    else
	min_y_bound_dist = abs(kp.y-rect.y-rect.height);
    return std::min(min_x_bound_dist,min_y_bound_dist);
}