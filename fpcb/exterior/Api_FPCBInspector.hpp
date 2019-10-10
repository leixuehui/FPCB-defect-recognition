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
	// ����������
	// Input:
	//		matchData	�ֶ�λ��������
	//		isTop		�Ƿ�Ϊ����
	// Output:
	// 		NULL
	/**********************************************/
	int api_init(const CoarseMatchData &matchData, int isTop);

	//����matchData�ļ�(������汾)
	int api_init(string matchJson);

    /**********************************************/
    // api_getVersionNum
    // ����������
    // Input:
    //		NULL
    // Output:
    // 		n   �汾��
    /**********************************************/
    static void api_getVersionNum(string &versionNum);
	/**********************************************/
	// api_init_ImageNum
	// ͼ�������ʼ��
	// Input:
	//		NULL
	// Output:
	// 		NULL
	/**********************************************/
	int api_init_ImageNum();

	/**********************************************/
	// api_loadParam
	// ���볬������д���ļ���·��
	// Input:
	//		dir		���������ļ���
	// Output:
	// 		NULL
	/**********************************************/
	int api_loadParam(string dir);
    
	/**********************************************/
	// api_destory
	// �ͷŶ���
	// Input:
	//		NULL
	// Output:
	// 		NULL
	/**********************************************/
	int api_destory();

	//==========================learn=========================//
	
	/**********************************************/
	// api_learn
	// ѵ��ģʽ(����Ĭ�϶�λģʽ)
	// Input:
	//		learnImage	{ͼ��ͷ��ַ����ȣ��߶ȣ���ȣ�ͼ��λ��}
	//		isTop		�Ƿ�Ϊ����	
	// Output:
	// 		NULL
	/**********************************************/
	int api_learn(ImageInfo &learnImage, int isTop);

    /**********************************************/
    // api_set_configparam
    // �ⲿ����������
    // Input:
    //		configParam    //�������
    //      isTop          //��������
    // Output:
    // 		NULL
    /**********************************************/
    int api_set_configParam(const ConfigParam &configParam, int isTop);

    /**********************************************/
    // api_get_configparam
    // �ⲿ��ȡ�������
    // Input:
    //      isTop          //��������
    // Output:
    //		configParam    //�������
    /**********************************************/
    int api_get_configParam(ConfigParam &configParam, int isTop);

	//==========================process=======================//

	/**********************************************/
	// api_save_param
	// ���ؽ�������������ռ�
	// Input:
	//		outputSpace	���������ռ�
	// Output:
	// 		NULL		
	/**********************************************/
	int api_load_space(OutputInfo *outputSpace);

	/**********************************************/
	// api_process
	// �����ƴ��ģʽ(����Ĭ�϶�λģʽ����Ҫ��ִ��api_load_space)
	// Input:
	//		input		���ͼ������
	//		isTop		�Ƿ�Ϊ����
	// Output:
	// 		outputSpace	������д�������ռ���		
	/**********************************************/
    int api_process(ImageInfo &input, int isTop);

	/**********************************************/
	// api_process_defects
	// ��ⲻƴ��ģʽ(����Ĭ�϶�λģʽ)
	// Input:
	//		input		���ͼ������
	//		isTop		�Ƿ�Ϊ����
	//		configParam	�������
	// Output:
	// 		defects		���ʵ��ͼȱ����Ϣ
	/**********************************************/
	int api_process_defects(ImageInfo &input, int isTop, ConfigParam & configParam, std::vector<DefectRoi>& defects);

	/**********************************************/
	// api_process_realtime
	// ���ټ��ģʽ�������ɶ�λ��ֻ����ǰ��λ����ͼƬ��
	// Input:
	//		configParam	�������
	// Output:
	// 		defects		���ʵ��ͼȱ����Ϣ
	/**********************************************/
	int api_process_realtime(ConfigParam &configParam, int isTop, std::vector<DefectRoi>& defects);
	
	/**********************************************/
	// api_location
	// ��λģʽ(��ѡ�����λ����ͼ)
	// Input:
	//		output		��λ����ͼ��ȡ�ռ�
	//		scale		���ű���
	//		brushWidth	��λ��Ե���ƿ��
	// Output:
	// 		NULL
	/**********************************************/
	int api_location(ImageInfo *output, float scale = 0.1f, int brushWidth = 1);

	/**********************************************/
	// api_gerber_location
	// gerber��Ⱦͼ��λ�������ʼ������ã�
	// Input:
    //      matchData   �ֶ�λ��������
	//		output		��Ⱦͼ����ռ�
	//		isTop		�Ƿ�Ϊ����
	//		lineWidth	(Ԥ���
	// Output:
	// 		NULL
	/**********************************************/
	int api_gerber_location(const CoarseMatchData &matchData, ImageInfo &output, int isTop, int lineWidth = 20);

	//==========================edit=======================//

    //==========================drawTool=======================//
    int api_processInterface(const ImageInfo &input, int imageIndex, int isTop, std::vector<DefectRoi>& defects);
    
    //ͳһ������
    int api_unifyConfigparam(int isTop, Layer layer);

    //��ȡֱ��ͼ
    int api_getHistogram(const ImageInfo &input, int x, int y, uchar lower[3], uchar upper[3], float ch1[256], float ch2[256], float ch3[256]);

    //ѵ����������
    int api_setLocalParam(uchar lower[3], uchar upper[3]);
    //==========================drawTool=======================//

	/**********************************************/
	// api_fillup
	// gerber��书��
	// Input:
	//		jsonList	����������η�Χ
	//		mouseX		������ӵ�x��
	//		mouseY		������ӵ�y��
	// Output:
	// 		filledX		���λ�ü�
	//		filledY		���λ�ü�
	/**********************************************/
	//int api_fillup(const path_t* jsonList, double mouseX, double mouseY, std::vector<double>& filledX, std::vector<double>& filledY);

	//========================calibration==================//
	/**********************************************/
	// api_doCalib
	// �궨����
	// Input:
	//		img				�궨ͼ��
	//		iIndex			�궨λ�ã��궨�����Ű������λ�ã�
	//		szMachineName	��̨��
	//		fAxisOffset		��̨x��ƫ����	
	// Output:
	// 		����궨�ļ�
	/**********************************************/
	//int api_doCalib(const ImageInfo &img, int iIndex, std::string szMachineName, float fAxisOffset);

	/**********************************************/
	// api_breakup
	// gerber�����и�
	// Input:
	//		jsonList		������ѡ�е�item
	//		mask			����ѡ��ROI
	// Output:
	// 		ploygon			���صĶ��������
	/**********************************************/
	//int api_breakup(const path_t * jsonList, const std::vector<bm::base::bmShapePoint>& mask, std::vector<std::vector<bm::base::bmShapePoint>>& ploygon);

	/**********************************************/
	// api_getHist
	// ��ȡƽ��ֱ��ͼ
	// Input:
	//		NULL
	// Output:
	// 		hist			ֱ��ͼ��
	/**********************************************/
	int api_getHist(ItemHyperparam &hist, int isTop);

	/**********************************************/
	// extraGoldPad(ͣ�ã�ת������dll��)
	// ��ȡ����
	// Input:
	//		line			��·����Ϣ
	//		baojiao			��������Ϣ
	// Output:
	// 		vecContour		���̲���Ϣ
	/**********************************************/
	//int api_extraGoldPad(const path_t* line, const path_t* baojiao, std::vector<std::vector<bm::base::bmShapePoint>>& vecContour);

    /**********************************************/
    // api_getCalibData
    // ��ȡ�궨����
    // Input:
    //		szMachineName   ��������
    // Output:
    // 		data		    �궨����
    /**********************************************/
    //int api_getCalibData(std::string szMachineName, CalibData& data);

    /**********************************************/
    // �ӿ����ƣ� observeFirstMark
    // ����˵��:  mark�����
    // �������:
    //	@coarseData_input	�����ʼ������
    //	@isTop				�Ƿ�Ϊ�����
    //	@firstSubImg_img	��0����Ұͼ
    //  @extend				mark��������Χ
    //     ...
    // �������:
    //	@firstSubImg_output	�����0����Ұͼ��������Ӧλ�û���mark
    //     ...
    // ����ֵ:
    //		0	����
    //	other	����
    //	   ...
    // ����/����: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_observeFirstMark(
        const CoarseMatchData &coarseData_input, int isTop, 
        const ImageInfo& firstSubImg_input, ImageInfo& firstSubImg_output, int extend = 300);

    /**********************************************/
    // �ӿ����ƣ� setTestImage
    // ����˵��:  ��ŵ�ǰ����ͼƬ
    // �������:
    //	@src		����ͼ��
    //	@iImgIndex	����ͼ�����Ұ���
    //	@isTop		�Ƿ�Ϊ�����
    //     ...
    // �������:
    //     ...
    // ����ֵ:
    //		0	����
    //	other	����
    //	   ...
    // ����/����: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_setTestImage(ImageInfo src, int iImgIndex, int isTop);

    /**********************************************/
    // �ӿ����ƣ� getTestGerbCoarse
    // ����˵��:  ��ȡ��·��gerb����
    // �������:
    //     ...
    // �������:
    //	@lays	���gerb����
    //     ...
    // ����ֵ:
    //		0	����
    //	other	����
    //	   ...
    // ����/����: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_getTestGerbCoarse(std::vector<PolyLay>&, int isTop);

    /**********************************************/
    // �ӿ����ƣ� setTestPosition
    // ����˵��:  �����û�����ק�Ľ��
    // �������:
    // @gerbLTpts	gerb���Ͻǵĵ�
    // @gerbRBpts	gerb���½ǵĵ�
    // @imgLTpts	img���Ͻǵĵ�
    // @imgRBpts	img���½ǵĵ�
    //     ...
    // �������:
    //     ...
    // ����ֵ:
    //		0	����
    //	other	����
    //	   ...
    // ����/����: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_setTestPosition(
        std::vector<bm::base::bmShapePoint> gerbLTpts, std::vector<bm::base::bmShapePoint> gerbRBpts,
        std::vector<bm::base::bmShapePoint> imgLTpts, std::vector<bm::base::bmShapePoint> imgRBpts,
        int isTop);

    /**********************************************/
    // �ӿ����ƣ� getTestGerbModify
    // ����˵��:  ��ȡ�޸ĵ������gerb����
    // �������:
    //     ...
    // �������:
    //@lays		���gerb����
    //     ...
    // ����ֵ:
    //		0	����
    //	other	����
    //	   ...
    // ����/����: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_getTestGerbModify(std::vector<PolyLay>&, int isTop);

    /**********************************************/
    // �ӿ����ƣ�doTest 
    // ����˵��: ���ڲ���ͼ��ģ
    // �������:
    //     ...
    // �������:
    //     ...
    // ����ֵ:
    //		0	����
    //	other	����
    //	   ...
    // ����/����: Wan Xiaofeng/02/07/2018
    /**********************************************/
    int api_doTest(int isTop);
private:
    void *g_processor = nullptr;
};