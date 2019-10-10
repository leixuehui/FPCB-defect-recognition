#pragma once
#include <iostream>
#include <io.h>
//#include "../intra/segmentStruct.hpp"
#include <fpcbParam.hpp>

using namespace std;

#define FPCBINSPECTOR_EXPORTS

#ifdef FPCBINSPECTOR_EXPORTS
#define FPCBINSPECTORDLL_API __declspec(dllexport)
#else
#define FPCBINSPECTORDLL_API __declspec(dllimport)
#endif // FPCBINSPECTOR_EXPORTS

class FPCBINSPECTORDLL_API APIFPCBInspector
{
public:
	APIFPCBInspector();
    ~APIFPCBInspector();

	//==========================init==========================//

	/**********************************************/
	// api_init
	// 创建检测对象
	// Input:
	//		matchData	粗定位所需资料
	//		isTop		是否为正面
	// Output:
	// 		NULL
	/**********************************************/
	int api_init(const CoarseMatchData &matchData, int isTop);

	//加载matchData文件(非软件版本)
	int api_init(string matchJson);

    /**********************************************/
    // api_getVersionNum
    // 创建检测对象
    // Input:
    //		NULL
    // Output:
    // 		n   版本号
    /**********************************************/
    static void api_getVersionNum(string &versionNum);
	/**********************************************/
	// api_init_ImageNum
	// 图像计数初始化
	// Input:
	//		NULL
	// Output:
	// 		NULL
	/**********************************************/
	int api_init_ImageNum();

	/**********************************************/
	// api_loadParam
	// 载入超参数读写的文件夹路径
	// Input:
	//		dir		超参所在文件夹
	// Output:
	// 		NULL
	/**********************************************/
	int api_loadParam(string dir);
    
	/**********************************************/
	// api_destory
	// 释放对象
	// Input:
	//		NULL
	// Output:
	// 		NULL
	/**********************************************/
	int api_destory();

	//==========================learn=========================//
	
	/**********************************************/
	// api_learn
	// 训练模式(集成默认定位模式)
	// Input:
	//		learnImage	{图像头地址，宽度，高度，跨度，图像位置}
	//		isTop		是否为正面	
	// Output:
	// 		NULL
	/**********************************************/
	int api_learn(ImageInfo &learnImage, int isTop);

    /**********************************************/
    // api_set_configparam
    // 外部导入界面参数
    // Input:
    //		configParam    //界面参数
    //      isTop          //正反面标记
    // Output:
    // 		NULL
    /**********************************************/
    int api_set_configParam(const ConfigParam &configParam, int isTop);

    /**********************************************/
    // api_get_configparam
    // 外部获取界面参数
    // Input:
    //      isTop          //正反面标记
    // Output:
    //		configParam    //界面参数
    /**********************************************/
    int api_get_configParam(ConfigParam &configParam, int isTop);

	//==========================process=======================//

	/**********************************************/
	// api_save_param
	// 加载界面参数与检测结果空间
	// Input:
	//		outputSpace	检测结果保存空间
	// Output:
	// 		NULL		
	/**********************************************/
	int api_load_space(OutputInfo *outputSpace);

	/**********************************************/
	// api_process
	// 检测且拼接模式(集成默认定位模式，需要先执行api_load_space)
	// Input:
	//		input		检测图像数据
	//		isTop		是否为正面
	// Output:
	// 		outputSpace	输出结果写入检测结果空间中		
	/**********************************************/
    int api_process(ImageInfo &input, int isTop);

	/**********************************************/
	// api_process_defects
	// 检测不拼接模式(集成默认定位模式)
	// Input:
	//		input		检测图像数据
	//		isTop		是否为正面
	//		configParam	界面参数
	// Output:
	// 		defects		输出实物图缺陷信息
	/**********************************************/
	int api_process_defects(ImageInfo &input, int isTop, ConfigParam & configParam, std::vector<DefectRoi>& defects);

	/**********************************************/
	// api_process_realtime
	// 快速检测模式（不集成定位，只处理当前定位完后的图片）
	// Input:
	//		configParam	界面参数
	// Output:
	// 		defects		输出实物图缺陷信息
	/**********************************************/
	int api_process_realtime(ConfigParam &configParam, int isTop, std::vector<DefectRoi>& defects);
	
	/**********************************************/
	// api_location
	// 定位模式(可选输出定位缩略图)
	// Input:
	//		output		定位缩略图提取空间
	//		scale		缩放倍数
	//		brushWidth	定位边缘绘制宽度
	// Output:
	// 		NULL
	/**********************************************/
	int api_location(ImageInfo *output, float scale = 0.1f, int brushWidth = 1);

	/**********************************************/
	// api_gerber_location
	// gerber渲染图定位输出（初始化后调用）
	// Input:
    //      matchData   粗定位所需资料
	//		output		渲染图输出空间
	//		isTop		是否为正面
	//		lineWidth	(预留项）
	// Output:
	// 		NULL
	/**********************************************/
	int api_gerber_location(const CoarseMatchData &matchData, ImageInfo &output, int isTop, int lineWidth = 20);

	//==========================edit=======================//

    //==========================drawTool=======================//
    int api_processInterface(const ImageInfo &input, int imageIndex, int isTop, std::vector<DefectRoi>& defects);
    
    //统一检测参数
    int api_unifyConfigparam(int isTop, Layer layer);

    //获取直方图
    int api_getHistogram(const ImageInfo &input, int x, int y, uchar lower[3], uchar upper[3], float ch1[256], float ch2[256], float ch3[256]);

    //训练参数交互
    int api_setLocalParam(uchar lower[3], uchar upper[3]);
    //==========================drawTool=======================//

	/**********************************************/
	// api_fillup
	// gerber填充功能
	// Input:
	//		jsonList	填充区域多边形范围
	//		mouseX		填充种子点x轴
	//		mouseY		填充种子点y轴
	// Output:
	// 		filledX		填充位置集
	//		filledY		填充位置集
	/**********************************************/
	//int api_fillup(const path_t* jsonList, double mouseX, double mouseY, std::vector<double>& filledX, std::vector<double>& filledY);

	//========================calibration==================//
	/**********************************************/
	// api_doCalib
	// 标定工具
	// Input:
	//		img				标定图像
	//		iIndex			标定位置（标定完整张板的所有位置）
	//		szMachineName	机台名
	//		fAxisOffset		机台x轴偏移量	
	// Output:
	// 		保存标定文件
	/**********************************************/
	//int api_doCalib(const ImageInfo &img, int iIndex, std::string szMachineName, float fAxisOffset);

	/**********************************************/
	// api_breakup
	// gerber数据切割
	// Input:
	//		jsonList		界面上选中的item
	//		mask			鼠标框选的ROI
	// Output:
	// 		ploygon			返回的多边形轮廓
	/**********************************************/
	//int api_breakup(const path_t * jsonList, const std::vector<bm::base::bmShapePoint>& mask, std::vector<std::vector<bm::base::bmShapePoint>>& ploygon);

	/**********************************************/
	// api_getHist
	// 获取平均直方图
	// Input:
	//		NULL
	// Output:
	// 		hist			直方图集
	/**********************************************/
	int api_getHist(ItemHyperparam &hist, int isTop);

	/**********************************************/
	// extraGoldPad(停用，转到工具dll上)
	// 抽取焊盘
	// Input:
	//		line			线路层信息
	//		baojiao			保胶层信息
	// Output:
	// 		vecContour		焊盘层信息
	/**********************************************/
	//int api_extraGoldPad(const path_t* line, const path_t* baojiao, std::vector<std::vector<bm::base::bmShapePoint>>& vecContour);

    /**********************************************/
    // api_getCalibData
    // 抽取标定数据
    // Input:
    //		szMachineName   机器名称
    // Output:
    // 		data		    标定数据
    /**********************************************/
    //int api_getCalibData(std::string szMachineName, CalibData& data);

    /**********************************************/
    // 接口名称： observeFirstMark
    // 功能说明:  mark点查找
    // 输入参数:
    //	@coarseData_input	输入初始化数据
    //	@isTop				是否为正面板
    //	@firstSubImg_img	第0个视野图
    //  @extend				mark的搜索范围
    //     ...
    // 输出参数:
    //	@firstSubImg_output	输出第0个视野图，并在相应位置绘制mark
    //     ...
    // 返回值:
    //		0	正常
    //	other	错误
    //	   ...
    // 作者/日期: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_observeFirstMark(
        const CoarseMatchData &coarseData_input, int isTop, 
        const ImageInfo& firstSubImg_input, ImageInfo& firstSubImg_output, int extend = 300);

    /**********************************************/
    // 接口名称： setTestImage
    // 功能说明:  存放当前测试图片
    // 输入参数:
    //	@src		输入图像
    //	@iImgIndex	输入图像的视野编号
    //	@isTop		是否为正面板
    //     ...
    // 输出参数:
    //     ...
    // 返回值:
    //		0	正常
    //	other	错误
    //	   ...
    // 作者/日期: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_setTestImage(ImageInfo src, int iImgIndex, int isTop);

    /**********************************************/
    // 接口名称： getTestGerbCoarse
    // 功能说明:  获取线路层gerb数据
    // 输入参数:
    //     ...
    // 输出参数:
    //	@lays	输出gerb数据
    //     ...
    // 返回值:
    //		0	正常
    //	other	错误
    //	   ...
    // 作者/日期: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_getTestGerbCoarse(std::vector<PolyLay>&, int isTop);

    /**********************************************/
    // 接口名称： setTestPosition
    // 功能说明:  设置用户拖拉拽的结果
    // 输入参数:
    // @gerbLTpts	gerb左上角的点
    // @gerbRBpts	gerb右下角的点
    // @imgLTpts	img左上角的点
    // @imgRBpts	img右下角的点
    //     ...
    // 输出参数:
    //     ...
    // 返回值:
    //		0	正常
    //	other	错误
    //	   ...
    // 作者/日期: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_setTestPosition(
        std::vector<bm::base::bmShapePoint> gerbLTpts, std::vector<bm::base::bmShapePoint> gerbRBpts,
        std::vector<bm::base::bmShapePoint> imgLTpts, std::vector<bm::base::bmShapePoint> imgRBpts,
        int isTop);

    /**********************************************/
    // 接口名称： getTestGerbModify
    // 功能说明:  获取修改调整后的gerb数据
    // 输入参数:
    //     ...
    // 输出参数:
    //@lays		输出gerb数据
    //     ...
    // 返回值:
    //		0	正常
    //	other	错误
    //	   ...
    // 作者/日期: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_getTestGerbModify(std::vector<PolyLay>&, int isTop);

    /**********************************************/
    // 接口名称：doTest 
    // 功能说明: 基于测试图像建模
    // 输入参数:
    //     ...
    // 输出参数:
    //     ...
    // 返回值:
    //		0	正常
    //	other	错误
    //	   ...
    // 作者/日期: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_doTest(int isTop);
private:
    void *g_processor = nullptr;
};