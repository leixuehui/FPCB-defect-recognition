#pragma once
#include <memory>
#include "detectBase.hpp"
#include <Utility.h>
#include <SegmentParts_Dll.h>
#include "threadPool.hpp"

struct OBJSELECTED
{
    int isTop = 0;
    int imageIdx = 0;
    int pcsIdx = 0;
    int itemIdx = 0;
    Layer layer;
    bool getStatus = false;
};

///////////////////////////////////////////////////////////////////////////
//inspector
///////////////////////////////////////////////////////////////////////////
class Inspector
{
public:
	Inspector();
	~Inspector();

	int init(const CoarseMatchData_Input matchData, int isTop);

	int destory();

	int loadParam(std::string dir);

	//==========================learn=========================//
	int learn(cv::Mat &learnImage, int imageIndex, int isTop);

    //内部加载正反面参数
	int setConfigparam(const ConfigParam &configParam, int isTop);

    //内部导出正反面参数
	int getConfigparam(ConfigParam &configParam, int isTop);
    
	//==========================process=======================//
	//载入检测结果空间
	int loadSpace(OutputInfo *outputSpace);
	//检测模式
	int process(cv::Mat &input, int imageIndex, int isTop);

    //交互检测模式
    int processInterface(const cv::Mat &input, int imageIndex, int isTop, std::vector<DefectRoi>& defects, bool isChange = false);

	int process(cv::Mat &input, int imageIndex, int isTop, ConfigParam &configParam, std::vector<DefectRoi>& defects);

	//实时检测模式
	int processRealtime(ConfigParam &configParam,int isTop, std::vector<DefectRoi>& defects);

	//gerber渲染图定位输出
	//int gerberLocation(cv::Mat &output, int lineWidth, int isTop);

	int location(cv::Mat &output, int imageNumber, float scale, int brushWidth = 1);

	int getHists(ItemHyperparam &hist, int isTop);

    //=========================interface tool======================
    //统一检测参数
    int unifyConfigparam(int isTop, Layer layer);

    //获取直方图
    int getHistogram(const cv::Mat &image, int x, int y, uchar lower[3], uchar upper[3], float ch1[256], float ch2[256], float ch3[256]);

    //训练参数交互
    int setLocalParam(uchar lower[3], uchar upper[3]);

private:
	//===========================xml=========================//
	int readHyperParam(std::string path, HyperParam *param, int isTop);
	int writeHyperParam(std::string path, HyperParam *param, int isTop);
    //==========================xml V2=======================//
    int writeXmlParam(std::string path, const TrainParam &trainParam, int isTop, int nImage);
    int readXmlParam(std::string path);
    
    
    int epmFilter(const cv::Mat &src, cv::Mat &dst, int radius, float delta);
	int saveInputParam(const std::string &dir, const int isTop);//调试和查看输入参数时用的；2019-08-13；

    int viewItems(cv::Mat & output, const ConfigParam &configParam, int imageNumber, float scale, int brushWidth);
    int visualItems(cv::Mat & output, const ConfigParam &configParam, const int imageNumber);
private:
	//signal
	int gTrainSignal = 0;
	int gProcessSignal = 0;
	//imageinfo
	int gImageWidth;
	int gImageHeight;
	int gImageNumber;
	int gAccImageNum;

	//opencv memory after imagemosaic
	std::vector<cv::Mat> gImageSpace;
	//3通道内存
	cv::Mat channels[3];
	unsigned char *gMemoryPool;
	//std::vector<cv::Mat> gImageMask;
	std::vector<std::vector<cv::Mat>> gImageMean;		//mean image;
	std::vector<std::vector<cv::Mat>> gImageStdDev;		//stamdard deviation image;

	ALL_REGION *gItems;									//all the item
	ConfigParam gParam;									//config
	ThreadPool *threadPool = nullptr;
	HyperParam gHyperParam;								//type hyper-parameter;
    
    TrainParam gTrainParam, gTrainParamTop, gTrainParamBack;
    ConfigParam gParamTop, gParamBack;

    std::vector<DefectRoi> defectsForInterface;
    OBJSELECTED itemSelected;

	bool		bPosParam = false;
	bool		bNegParam = false;

	//configure directory
	std::string gConfigPaths;

	int gMaxPcsWidth, gMaxPcsHeight;
public:
	cv::Mat	gImageSrc;									//region image;

	HeterochrosisParam gHeterParam;
	OutputInfo *gOutputInfo;
};