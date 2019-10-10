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
//��־�ļ�
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

#define BASEWIN 100     //100���صĻ���

#define PIXEL_NORMALIZATION 255

#define TRAIN_SPACE 32

struct Items
{
	cv::Mat	image;	//����ͼ��
	cv::Mat mask;	//��ģͼ��
	double	phi;	//��ת�Ƕ�
	Layer	type;	//��������
	int		tab;	//�������
	int		nTab;	//��������
};

//��ⷽ��
enum Method
{
	SimpleMethod,   //�򵥱���ͼ��
    ComplexMethod,  //���ӱ���ͼ��
    SpecialMethod   //�ض�����
};

struct MeanStd
{
    int area = 0;
    float mean[3] = {0.f};
    float stddev[3] = {0.f};
    float upper[3] = {3.f, 3.f, 3.f};
    float lower[3] = {3.f, 3.f, 3.f};
    bool unify = true; //Ĭ��Ϊͳһ����,Ϊ�����û��������; ��ͳһΪ����ѵ����Ӧ�Ĳ���
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

//ѵ������ver.2.
class TrainParam
{
public:
    int nSimple = 0;                                           //��ѵ��layer�������

    int nComplex = 0;                                          //����ѵ��layer�������

    int maxPcsWidth = 0;                                        //���PCS���

    int maxPcsHeight = 0;                                       //���PCS�߶�

    SimpleParam simpleParam[TRAIN_SPACE];                       //��ѵ������

    ComplexParam complexParam[TRAIN_SPACE];                     //����ѵ������

    inline int getSimpleIndex(Layer layer)const                 //��ѵ��layer��λ
    {
        for (int i = 0; i < TRAIN_SPACE; i++)
        {
            if(layer == simpleParam[i].layer)   return i;
        }
        return -1;
    }

    inline int getComplexParam(Layer layer)const                //����ѵ��layer��λ
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

//������ʱ�ռ����
struct SpaceParam
{
	uchar *kernelSpace = NULL;			//��̬ѧ��
	int kernelSpaceTotalWidth = 0;		//��̬ѧ�˿�����

	uchar *pcsSpace = NULL;				//pcs���ռ�(pcs * cpu thread number)
	int pcsSpaceTotalWidth = 0;			//pcs���ռ������
	
	uchar *imageSpace = NULL;			//image���ռ�
	
	bool *memoryLock = NULL;			//�ڴ�״̬(һ�����룬�ù���)
	int MemorySize = 0;					//�ڴ�����
};

struct PadTrain_SGM 
{
	unsigned char *pMatrixTrain;
	unsigned int rowMatrix;
};

struct HyperParam
{
	PadParam			pad[256];		//���̿ռ�
	
	int					nPad = 0;		//������������

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
	//int area;					//ȱ�����
	std::vector<DftAbstract> roi;	//ȱ������
	int n;						//ȱ�����ڵ�ͼ��
	int nPcs;					//ȱ������PCS
    bool isCuda;                //�Ƿ�Ϊ���ѧϰ���
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
	
    //��ʼ�����ռ�
	static int baseInit(int threadNum, cv::Size pcsSize, cv::Size imageSize);
	static int trainInit();
	static int trainDestroy();

    //itemsѵ��
	static int itemTrainer(const std::string &savePath, const cv::Mat &image, const ALL_REGION &gItems,ConfigParam &configParam,HyperParam &hyperParam,int &maxPcsWidth,int &maxPcsHeight/*const TrainInput &input, TrainOutput *output*/);

    //items ѵ��V2
    static int itemTrainerV2(const cv::Mat &image, const int &isTop, const int nImage, const ALL_REGION & gItems, TrainParam &param);

    //items���
	static int itemInspector(cv::Mat *img, int isTop, const PCS_REGION &gItems, const ConfigParam &configParam, DefectInfo &defectInfo, bool isModify = false);

    static int itemInspectorV2(const cv::Mat &img, const int &isTop, const int &nImage, const PCS_REGION &gItems, TrainParam &param, const ConfigParam &configParam, DefectInfo &defectInfo);

    //���ѧϰ��ʼ��
    static int cudaInit();

    //items���(���ѧϰ�汾)
    static int itemInspectorCuda(cv::Mat *img, const ALL_REGION &gItems, ConfigParam configParam, std::vector<DefectCudaInfo> &defectInfo);
	/*static int itemInspectorCuda(cv::Mat *img, const ALL_REGION &gItems, ConfigParam configParam, std::vector<DefectInfo> &defectInfo);*/
    //items����鿴
	static int itemView(std::string savePath, const cv::Mat &image, ALL_REGION &gItems, HyperParam &hyperParam);

	static void maskFitting(const cv::Mat &obj, cv::Mat &mask);

	//���ؼ�����
	static void loadParam(const HyperParam &param, int isTop);

	//����������
	static void emitParam(HyperParam &param, int isTop);

//�������㷨����
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
    
    //��ֿɼ���
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
	static ConfigParam gLastConfigParam;	//�洢���ı�Ĳ����������ж����²����仯
	static HyperParam gHyperParamFront;		//����׼��������
	static HyperParam gHyperParamBack;		//����׼��������
	static HyperParam gHyperParam;
	//������ʱ�ռ����
	static SpaceParam gSpaceParam;
    //ӳ�����
    static cv::Mat gLut,areaLut;
	//PAD_SGM����Ŀռ䣻
	static PadTrain_SGM gMatrixPadTrain[256];

#ifndef _DEBUG
    static PyObject *pModule;
    static PyObject *pInit;             //ģ���ʼ��
    static PyObject *pInitObj;          //ʵ��������
    static PyObject *pInfer_images;     //��⺯��
    static PyObject *arr_images_append; //����ͼ��
    static PyObject *arr_index_append;  //����pcs��
#endif // !_DEBUG



};