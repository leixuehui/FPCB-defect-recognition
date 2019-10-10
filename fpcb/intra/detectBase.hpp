#pragma once
#include <Windows.h>
#include <vector>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>

#include <segmentStruct.h>
#include <fpcbParam.hpp>
//#include <json.hpp>
#include "threadPool.hpp"
//日志文件
#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifndef _DEBUG
    #include <Python.h>
    #include <numpy/arrayobject.h>
#endif // !_DEBUG

//using namespace cv;
//using namespace std;
//using namespace nlohmann;


#define _TIME_LOG_
#define _PARAM_LOG_

#ifdef _TIME_LOG_
extern std::string log_time_name ;

#endif

#ifdef _PARAM_LOG_
extern std::string log_param_name;
#endif


extern void iniLogAlg();

#define WIDTH 5000
#define HEIGHT 5000

#define BASEWIN 100     //100像素的滑窗

#define PIXEL_NORMALIZATION 255

#define TRAIN_SPACE 32

struct Items
{
	cv::Mat	image;	//样本图像
	cv::Mat mask;	//掩模图像
	double	phi;	//旋转角度
	Layer	type;	//样本类型
	int		tab;	//样本标号
	int		nTab;	//类型总数
};

//检测方案
enum Method
{
	SimpleMethod,   //简单背景图像
    ComplexMethod,  //复杂背景图像
    SpecialMethod   //特定方案
};

struct MeanStd
{
    int area = 0;
    float mean[3] = {0.f};
    float stddev[3] = {0.f};
    float upper[3] = {3.f, 3.f, 3.f};
    float lower[3] = {3.f, 3.f, 3.f};
    bool unify = true; //默认为统一参数,为采用用户输入参数; 不统一为采用训练对应的参数
};

struct SimpleParam
{
    Layer layer;
    MeanStd data[256];
    int nItem = 0;
};

struct ComplexParam
{
    Layer layer;

};

//训练参数ver.2.
class TrainParam
{
public:
    int nSimple = 0;                                           //简单训练layer种类个数

    int nComplex = 0;                                          //复杂训练layer种类个数

    int maxPcsWidth = 0;                                        //最大PCS宽度

    int maxPcsHeight = 0;                                       //最大PCS高度

    SimpleParam simpleParam[TRAIN_SPACE];                       //简单训练参数

    ComplexParam complexParam[TRAIN_SPACE];                     //复杂训练参数

    inline int getSimpleIndex(Layer layer)const                 //简单训练layer定位
    {
        for (int i = 0; i < TRAIN_SPACE; i++)
        {
            if(layer == simpleParam[i].layer)   return i;
        }
        return -1;
    }

    inline int getComplexParam(Layer layer)const                //复杂训练layer定位
    {
        for (int i = 0; i < TRAIN_SPACE; i++)
        {
            if(layer == complexParam[i].layer)  return i;
        }
        return -1;
    }
    
    inline bool empty()
    {
        if (nSimple == 0 && nComplex == 0)  return true;
        
        return false;
    }
};

//申请临时空间参数
struct SpaceParam
{
	uchar *kernelSpace = NULL;			//形态学核
	int kernelSpaceTotalWidth = 0;		//形态学核宽限制

	uchar *pcsSpace = NULL;				//pcs检测空间(pcs * cpu thread number)
	int pcsSpaceTotalWidth = 0;			//pcs检测空间宽限制
	
	uchar *imageSpace = NULL;			//image检测空间
	
	bool *memoryLock = NULL;			//内存状态(一起申请，好管理)
	int MemorySize = 0;					//内存总数
};

struct PadTrain_SGM 
{
	unsigned char *pMatrixTrain;
	unsigned int rowMatrix;
};

struct HyperParam
{
	PadParam			pad[256];		//焊盘空间
	
	int					nPad = 0;		//焊盘种类数量

	SteelParam			steel[16];
	int					nSteel = 0;

	OpacityParam		opacity[16];
	int					nOpacity = 0;

	TransparencyParam	transparency[256];
	int                 nTransparency;

	CharParam			charactor;
	
	HoleParam			hole[256];
	int					nHole;

	LineParam			line;
	
	FingerParam			finger;

    CarveParam          carve[256];
    int                 nCarve;

};

struct DefectCuda
{
	int nImg;
	int nPcs;
	Layer abstract;
	int nID;
	cv::Rect roi;
};

struct DefectCudaInfo
{
	std::vector<DefectCuda> dft;
};

struct DFTINFO
{
    cv::Rect rect;
    int id;
};

struct DftAbstract
{
	std::vector<DFTINFO> info;
	Layer   abstract;
};

struct DefectInfo
{
	//int area;					//缺陷面积
	std::vector<DftAbstract> roi;	//缺陷区域
	int n;						//缺陷所在的图像
	int nPcs;					//缺陷所在PCS
    bool isCuda;                //是否为深度学习结果
};

struct InspectParam
{
	std::vector<cv::Mat> srcImage;
	cv::Mat *imageSpac;
	cv::Mat *imageMask;
	std::vector<std::vector<cv::Mat>> imageMean;
	std::vector<std::vector<cv::Mat>> imageStdDev;
};


///////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////

static std::mutex muMemMan, muDefWrite, muImg, muRoi, muCuda;
class AlgBase
{
public:
	AlgBase();
	~AlgBase();
	
    //初始化检测空间
	static int baseInit(int threadNum, cv::Size pcsSize, cv::Size imageSize);
	static int trainInit();
	static int trainDestroy();

    //items训练
	static int itemTrainer(const std::string &savePath, const cv::Mat &image, const ALL_REGION &gItems,ConfigParam &configParam,HyperParam &hyperParam,int &maxPcsWidth,int &maxPcsHeight/*const TrainInput &input, TrainOutput *output*/);

    //items 训练V2
    static int itemTrainerV2(const cv::Mat &image, const int &isTop, const int nImage, const ALL_REGION & gItems, TrainParam &param);

    //items检测
	static int itemInspector(cv::Mat *img, int isTop, const PCS_REGION &gItems, const ConfigParam &configParam, DefectInfo &defectInfo, bool isModify = false);

    static int itemInspectorV2(const cv::Mat &img, const int &isTop, const int &nImage, const PCS_REGION &gItems, TrainParam &param, const ConfigParam &configParam, DefectInfo &defectInfo);

    //深度学习初始化
    static int cudaInit();

    //items检测(深度学习版本)
    static int itemInspectorCuda(cv::Mat *img, const ALL_REGION &gItems, ConfigParam configParam, std::vector<DefectCudaInfo> &defectInfo);
	/*static int itemInspectorCuda(cv::Mat *img, const ALL_REGION &gItems, ConfigParam configParam, std::vector<DefectInfo> &defectInfo);*/
    //items输出查看
	static int itemView(std::string savePath, const cv::Mat &image, ALL_REGION &gItems, HyperParam &hyperParam);

	static void maskFitting(const cv::Mat &obj, cv::Mat &mask);

	//加载检测参数
	static void loadParam(const HyperParam &param, int isTop);

	//传出检测参数
	static void emitParam(HyperParam &param, int isTop);

//各类检测算法方法
private:
	//////////////////////////////////Trainer////////////////////////////////////

 #pragma region Train

	static int padTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img,  PadParam *pads, Layer layer, int &nPad);

	static int loadImageData(const ABSTRACT_REGIONS & region, 
		cv::Mat * img, std::vector<std::vector<cv::Mat>>& vecImg, 
		std::vector<std::vector<cv::Mat>>& vecMask, std::vector<unsigned int>& vecSumPix);

	static int padTrainer_SGM(const ABSTRACT_REGIONS &region, 
		cv::Mat *img, PadParam *pads, 
		int &nPad,
		PadTrain_SGM *matrix);

	static int steelTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img, SteelParam *steel, Layer layer, int &nSteel);

	static int opacityTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img,  OpacityParam *opacity, Layer layer, int &nOpacity);

	static int transprencyTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img, TransparencyParam *transparency, Layer layer, int pcsId, int &nTransprency);

    static int transprencyTrainerV2(const ABSTRACT_REGIONS &region, cv::Mat *img, TransparencyParam *transparency, Layer layer, int pcsId, int &nTransprency);

	static int lineTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img, LineParam *line);

	static int fingerTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img, FingerParam *finger);

	static int holeTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img, HoleParam *hole, int &nHole);

    static int carveTrainer(const ABSTRACT_REGIONS &region, cv::Mat *img, CarveParam *carve, Layer layer, int &nCarve);

#pragma endregion


#pragma region TrainV2
    static int simpleTrain(const ABSTRACT_REGIONS &region, const cv::Mat &img, TrainParam &param, Layer layer);

    static int complexTrain(const ABSTRACT_REGIONS &region, const cv::Mat &img, TrainParam &param, Layer layer);
    
    static int padTrainerV2(const ABSTRACT_REGIONS &region, cv::Mat *img, PadParam *pads, int &nPad);
#pragma endregion

#pragma region InspectorV2
    
    //拆分可加速
    static int simpleInspector(const ABSTRACT_REGIONS &region, const cv::Mat &img, const ConfigParam &configParam, TrainParam &param, Layer layer, std::vector<DFTINFO>& defectInfo);

    static int padInspectorV2(const ABSTRACT_REGIONS &region, const cv::Mat &img, const ConfigParam &configParam, TrainParam &param, Layer layer, std::vector<DFTINFO>& defectInfo);

    static int steelInspectorV2(const ABSTRACT_REGIONS &region, const cv::Mat &img, const ConfigParam &configParam, TrainParam &param, Layer layer, std::vector<DFTINFO>& defectInfo);

    static int emiInspectorV2(const ABSTRACT_REGIONS &region, const cv::Mat &img, const ConfigParam &configParam, TrainParam &param, Layer layer, std::vector<DFTINFO>& defectInfo);

    static int carveInspectorV2(const ABSTRACT_REGIONS &region, const cv::Mat &img, const ConfigParam &configParam, TrainParam &param, Layer layer, std::vector<DFTINFO>& defectInfo);
    
    static int complexInspector(const ABSTRACT_REGIONS &region, const cv::Mat &img, const ConfigParam &configParam, TrainParam &param, Layer layer, std::vector<DFTINFO>& defectInfo);

#pragma endregion

	//////////////////////////////////Inspector////////////////////////////////////
#pragma region Inspector

	static int padInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int padInspector_SGM(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int steelInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int opacityInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int transprencyInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

    static int nestInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

    static int transprencyInspectorV2(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int lineInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int figureInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int holeInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static int charInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

    static int carveInspector(const ABSTRACT_REGIONS &region, const ConfigParam &configParam, cv::Mat *img, std::vector<cv::Rect> &defectInfo, bool isModify = false);

	static void equalHistWithMask(cv::Mat src, float *normHist, cv::Mat dst, cv::Mat mask = cv::Mat());
	
	static float getWeightArea(const cv::Mat *channels, const cv::Rect &roi, const vec3f &tLowers, const vec3f &tUppers);

#pragma endregion


private:
	//static cv::Mat lut;
	static ConfigParam gLastConfigParam;	//存储最后改变的参数，用于判断最新参数变化
	static HyperParam gHyperParamFront;		//检测基准参数正面
	static HyperParam gHyperParamBack;		//检测基准参数反面
	static HyperParam gHyperParam;
	//申请临时空间参数
	static SpaceParam gSpaceParam;
    //映射矩阵
    static cv::Mat gLut,areaLut;
	//PAD_SGM所需的空间；
	static PadTrain_SGM gMatrixPadTrain[256];

#ifndef _DEBUG
    static PyObject *pModule;
    static PyObject *pInit;             //模块初始化
    static PyObject *pInitObj;          //实例化对象
    static PyObject *pInfer_images;     //检测函数
    static PyObject *arr_images_append; //载入图像
    static PyObject *arr_index_append;  //载入pcs号
#endif // !_DEBUG



};