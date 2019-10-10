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

    //�ڲ��������������
	int setConfigparam(const ConfigParam &configParam, int isTop);

    //�ڲ��������������
	int getConfigparam(ConfigParam &configParam, int isTop);
    
	//==========================process=======================//
	//���������ռ�
	int loadSpace(OutputInfo *outputSpace);
	//���ģʽ
	int process(cv::Mat &input, int imageIndex, int isTop);

    //�������ģʽ
    int processInterface(const cv::Mat &input, int imageIndex, int isTop, std::vector<DefectRoi>& defects, bool isChange = false);

	int process(cv::Mat &input, int imageIndex, int isTop, ConfigParam &configParam, std::vector<DefectRoi>& defects);

	//ʵʱ���ģʽ
	int processRealtime(ConfigParam &configParam,int isTop, std::vector<DefectRoi>& defects);

	//gerber��Ⱦͼ��λ���
	//int gerberLocation(cv::Mat &output, int lineWidth, int isTop);

	int location(cv::Mat &output, int imageNumber, float scale, int brushWidth = 1);

	int getHists(ItemHyperparam &hist, int isTop);

    //=========================interface tool======================
    //ͳһ������
    int unifyConfigparam(int isTop, Layer layer);

    //��ȡֱ��ͼ
    int getHistogram(const cv::Mat &image, int x, int y, uchar lower[3], uchar upper[3], float ch1[256], float ch2[256], float ch3[256]);

    //ѵ����������
    int setLocalParam(uchar lower[3], uchar upper[3]);

private:
	//===========================xml=========================//
	int readHyperParam(std::string path, HyperParam *param, int isTop);
	int writeHyperParam(std::string path, HyperParam *param, int isTop);
    //==========================xml V2=======================//
    int writeXmlParam(std::string path, const TrainParam &trainParam, int isTop, int nImage);
    int readXmlParam(std::string path);
    
    
    int epmFilter(const cv::Mat &src, cv::Mat &dst, int radius, float delta);
	int saveInputParam(const std::string &dir, const int isTop);//���ԺͲ鿴�������ʱ�õģ�2019-08-13��

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
	//3ͨ���ڴ�
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