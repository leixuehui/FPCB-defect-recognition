// Test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

//#include "pch.h"

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <io.h>
#include <typeinfo>
#include <Windows.h>
#include <regex>
#include <opencv2/opencv.hpp>


#include "../intra/threadPool.hpp"
#include <Utility.h>
//#include "../intra/segmentStruct.hpp"
#include "../exterior/Api_FPCBInspector.hpp"
#include <json-develop\single_include\nlohmann\json.hpp>
#include <SegmentParts_Dll.h>
#include <ParseJ.h>
#include "parsePath.h"

//#define CVUI_IMPLEMENTATION
//#include "cvui.h"

using namespace std;
using namespace cv;
using namespace nlohmann;

#define PI 3.1415926

/*
Mat getGaborFilter(float lambda, float theta,
	float sigma2, float gamma,
	float psi = 0.0f) {
	if (abs(lambda - 0.0f)<1e-6) {
		lambda = 1.0f;
	}
	float sigma_x = sigma2;
	float sigma_y = sigma2 / (gamma*gamma);
	int nstds = 3;
	float sqrt_sigma_x = sqrt(sigma_x);
	float sqrt_sigma_y = sqrt(sigma_y);
	int xmax = max(abs(nstds*sqrt_sigma_x*cos(theta)), abs(nstds*sqrt_sigma_y*sin(theta)));
	int ymax = max(abs(nstds*sqrt_sigma_x*sin(theta)), abs(nstds*sqrt_sigma_y*cos(theta)));
	int half_filter_size = xmax>ymax ? xmax : ymax;
	int filter_size = 2 * half_filter_size + 1;
	Mat gaber = Mat::zeros(filter_size, filter_size, CV_32F);
	for (int i = 0; i<filter_size; i++) {
		float* f = gaber.ptr<float>(i);
		for (int j = 0; j<filter_size; j++) {
			int x = j - half_filter_size;
			int y = i - half_filter_size;
			float x_theta = x*cos(theta) + y*sin(theta);
			float y_theta = -x*sin(theta) + y*cos(theta);
			f[x] = exp(-.5*(x_theta*x_theta / sigma_x + y_theta*y_theta / sigma_y));
			f[x] = f[x] * cos(2 * PI*x_theta / lambda + psi);
		};
	}
	return gaber;
}

Mat gaborFilter(Mat& img, Mat& filter) {
	int half_filter_size = (max(filter.rows, filter.cols) - 1) / 2;
	Mat filtered_img(img.rows, img.cols, CV_32F);
	for (int i = 0; i<img.rows; i++) {
		uchar* img_p = img.ptr<uchar>(i);
		float* img_f = filtered_img.ptr<float>(i);
		for (int j = 0; j<img.cols; j++) {
			float filter_value = 0.0f;
			for (int fi = 0; fi<filter.rows; fi++) {
				float* f = filter.ptr<float>(fi);
				int img_i = i + fi - half_filter_size;
				img_i = img_i < 0 ? 0 : img_i;
				img_i = img_i >= img.rows ? (img.rows - 1) : img_i;
				uchar* p = img.ptr<uchar>(img_i);
				for (int fj = 0; fj<filter.cols; fj++) {
					int img_j = j + fj - half_filter_size;
					img_j = img_j < 0 ? 0 : img_j;
					img_j = (img_j >= img.cols) ? (img.cols - 1) : img_j;
					float tmp = (float)p[img_j] * f[fj];
					filter_value += tmp;
				}
			}
			img_f[j] = filter_value;
		}
	}
	return filtered_img;
}

Mat normalizeFilterShow(Mat gaber) {
	Mat gaber_show = Mat::zeros(gaber.rows, gaber.cols, CV_8UC1);
	float gaber_max = FLT_MIN;
	float gaber_min = FLT_MAX;
	for (int i = 0; i<gaber.rows; i++) {
		float* f = gaber.ptr<float>(i);
		for (int j = 0; j<gaber.cols; j++) {
			if (f[j]>gaber_max) {
				gaber_max = f[j];
			}
			if (f[j]<gaber_min) {
				gaber_min = f[j];
			}
		}
	}
	float gaber_max_min = gaber_max - gaber_min;
	for (int i = 0; i<gaber_show.rows; i++) {
		uchar* p = gaber_show.ptr<uchar>(i);
		float* f = gaber.ptr<float>(i);
		for (int j = 0; j<gaber_show.cols; j++) {
			if (gaber_max_min != 0.0f) {
				float tmp = (f[j] - gaber_min)*255.0f / gaber_max_min;
				p[j] = (uchar)tmp;
			}
			else {
				p[j] = 255;
			}
		}
	}
	return gaber_show;
}

Mat gTmpImage32F, FS, FD, ED;

void waveletDecomposeHaar(Mat *image, Mat *waveletImage, int levelDepth)
{
	//当前层数
	int level = 1;
	float harrWeight = 0.35355339059327376220042218105242;
	float lowPassFilter[2] = { harrWeight, harrWeight };
	float highPassFilter[2] = { harrWeight, -harrWeight };
	
	//Mat FD, FS, ED;
	//FD = Mat::ones(gTmpImage32F.size(), CV_32FC1);
	//FS = Mat::ones(gTmpImage32F.size(), CV_32FC1);
	//ED = Mat::ones(gTmpImage32F.size(), CV_32FC1);
	for (int l = 0; l < levelDepth; l++)
	{
		int width = image->cols / level;
		int height = image->rows / level;

		//水平分解
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width / 2; c++)
			{
				if (2 * c + 1 >= width)	continue;
				for (int n = 0; n < image->channels(); n++)
				{
					//低通
					waveletImage->ptr<float>(r, c)[n] = (gTmpImage32F.ptr<float>(r, 2 * c)[n] + gTmpImage32F.ptr<float>(r, 2 * c + 1)[n]) * harrWeight;
					//高通
					waveletImage->ptr<float>(r, c + width / 2)[n] = (gTmpImage32F.ptr<float>(r, 2 * c)[n] - gTmpImage32F.ptr<float>(r, 2 * c + 1)[n]) * harrWeight;
				}
			}
		}
		//垂直分解
		for (int c = 0; c < width; c++)
		{
			for (int r = 0; r < height / 2; r++)
			{
				if (2 * r + 1 >= height)	continue;
				for (int n = 0; n < image->channels(); n++)
				{
					//低通
					gTmpImage32F.ptr<float>(r, c)[n] = (waveletImage->ptr<float>(2 * r, c)[n] + waveletImage->ptr<float>(2 * r + 1, c)[n]) * harrWeight;
					//高通
					gTmpImage32F.ptr<float>(r + height / 2, c)[n] = (waveletImage->ptr<float>(2 * r, c)[n] - waveletImage->ptr<float>(2 * r + 1, c)[n]) * harrWeight;
				}
			}
		}

		//获取低通

		for (int r = 0; r < height / 2; r++)
		{
			for (int c = 0; c < width / 2; c++)
			{
				float pixel[3] = { 0 };
				for (int h = -4; h <= 4; h++)
				{
					for (int w = -4; w <= 4; w++)
					{
						Point coord = { c + w, r + h };
						if (coord.x < 0 || coord.x >= width / 2 ||
							coord.y < 0 || coord.y >= height / 2)
						{
							pixel[0] += 0;
							pixel[1] += 0;
							pixel[2] += 0;
						}
						else
						{
							for (int n = 0; n < image->channels(); n++)
							{
								pixel[n] += pow(gTmpImage32F.ptr<float>(coord.y, coord.x)[n],2);
							}
						}
					}
				}
				for (int n = 0; n < image->channels(); n++)
				{
					//对齐
					FS.ptr<float>(r + height / 2, c + width / 2)[n] = pixel[n];
				}
			}
		}
		for (int r = height / 2; r < height; r++)
		{
			for (int c = width / 2; c < width; c++)
			{
				float pixel[3] = { 0 };
				for (int h = -4; h <= 4; h++)
				{
					for (int w = -4; w <= 4; w++)
					{
						Point coord = { c + w, r + h };
						if (coord.x < width / 2 || coord.x >= width ||
							coord.y < width / 2 || coord.y >= height)
						{
							pixel[0] += 0;
							pixel[1] += 0;
							pixel[2] += 0;
						}
						else
						{
							for (int n = 0; n < image->channels(); n++)
							{
								pixel[n] += pow(gTmpImage32F.ptr<float>(coord.y, coord.x)[n], 2);
							}
						}
					}
				}
				for (int n = 0; n < image->channels(); n++)
				{
					FD.ptr<float>(r, c)[n] = pixel[n];
				}
			}
		}

		//下采样
		level *= 2;
		//waveletImage->copyTo(gTmpImage32F);
	}
	gTmpImage32F.copyTo(*waveletImage);
	//提取对角分解的能量

	for (int r = 0; r < image->rows; r++)
	{
		for (int c = 0; c < image->cols; c++)
		{
			for (int n = 0; n < image->channels(); n++)
			{
				ED.ptr<float>(r, c)[n] = FD.ptr<float>(r, c)[n] / FS.ptr<float>(r, c)[n];
			}
		}
	}
}

int  testHeterBaseMean()
{
	std::string path = "..\\Data\\padregions";
	std::vector<std::pair<std::string, std::vector<int> > > vecPairFile2IndexPcs = {
		{ "1-L-0",{ 0,1,3,5,7,11,12,13,15 } },
		{ "1-R-0",{ 0,2,6,9,10 } },
		{ "2-L-0",{ 2,4,5,7,8,14,15,16,17 } },
		{ "2-R-0",{ 3,7,8,9,10,11 } },
		{ "3-L-0",{ 5,6,8,9,10,11,13,14,15 } },
		{ "3-R-0",{ 4,5,6,7,9,11 }},
		{ "1-L-1",{ 2,4,6,8,9,10,14,16 } },
		{ "1-R-1",{ 1,3,4,5,7,8 } },
		{ "2-L-1",{ 0,1,3,6,9,10,11,12,13 } },
		{ "2-R-1",{ 0,1,2,4,5,6 } },
		{ "3-L-1",{ 0,1,2,3,4,7,12 } },
		{ "3-R-1",{ 0,1,2,3,8,10 } }
	};
	size_t szVecPair = vecPairFile2IndexPcs.size();
	for (int i = 0; i < szVecPair; i++)
	{
		std::pair<std::string, std::vector<int>> pairTemp = vecPairFile2IndexPcs[i];
		std::string nameFile = pairTemp.first;
		std::vector<int> vecIndexPcs = pairTemp.second;

		size_t szIndexPcs = vecIndexPcs.size();
		for (int j = 0; j < szIndexPcs; j++)
		{
			int indexPcs = vecIndexPcs[j];
			for (int k = 0; k < 21; k++)
			{
				std::string nameRegion = path + "\\" + nameFile + "\\"
					+ std::to_string(indexPcs) + "-" + std::to_string(k) + ".bmp";
				cv::Mat imgSrc = cv::imread(nameRegion);

				std::string nameStandard = path + "\\standard\\standard_" + std::to_string(k) + ".bmp";
				cv::Mat imgMeanStandard = cv::imread(nameStandard);

				std::string nameStdDevStandard = path + "\\stddev\\stddev_" + std::to_string(k) + ".bmp";
				cv::Mat imgStdDevStandard = cv::imread(nameStdDevStandard);

				ImageInfo imageSrc, imageMean, imageStdDevStandard;
				imageSrc.heigth = imgSrc.rows;
				imageSrc.width = imgSrc.cols;
				imageSrc.ptr = imgSrc.data;
				imageSrc.step = imgSrc.step[0];

				InputInfo inputInfor;
				inputInfor.image = imageSrc;
				LayerInfo layerInfor;
				layerInfor.layer.priority = 1;
				inputInfor.layers.push_back(layerInfor);

				imageMean.heigth = imgMeanStandard.rows;
				imageMean.width = imgMeanStandard.cols;
				imageMean.ptr = imgMeanStandard.data;
				imageMean.step = imgMeanStandard.step[0];

				imageStdDevStandard.heigth = imgStdDevStandard.rows;
				imageStdDevStandard.width = imgStdDevStandard.cols;
				imageStdDevStandard.ptr = imgStdDevStandard.data;
				imageStdDevStandard.step = imgStdDevStandard.step[0];

				APIFPCBInspector inspector;
				//inspector.api_init("", 0, 0, 0, 0, 0);
				HeterochrosisParam heterParam;
				heterParam.minAreaConnection = 30;
				heterParam.tolerance = 4;
				heterParam.widthLeft = heterParam.widthBottom = heterParam.widthTop = heterParam.widthRight = 3;
				//inspector.api_setParam(&heterParam, &imageMean, &imageStdDevStandard);

				OutputInfo outputInfo;
				//inspector.api_process(&inputInfor, &outputInfo);

				inspector.api_destory();
				cv::waitKey(0);
			}
		}
	}
	return 0;
}

int testWavelet()
{
	Mat image = imread("../Data/5.jpg", 0);
	//blur(image, image, Size(5,5));
	//medianBlur(image, image, 5);
	Mat wave = Mat::zeros(image.size(), CV_32FC1);
	int width = image.cols;
	int height = image.rows;
	image.convertTo(gTmpImage32F, CV_32FC1);
	gTmpImage32F /= 255;
	FS = Mat::zeros(gTmpImage32F.size(), CV_32FC1);
	FD = Mat::zeros(gTmpImage32F.size(), CV_32FC1);
	ED = Mat::zeros(gTmpImage32F.size(), CV_32FC1);
	//Mat gabor = getGaborFilter(0.3, 0, 4, 2);

	double Time = (double)cvGetTickCount();
	//Mat gaborImage = gaborFilter(image, gabor);
	waveletDecomposeHaar(&image, &wave, 3);

	Mat deltaF2, deltaF1, deltaF3, C2, C3, S, E1, E2, E3;
	deltaF1 = FD(Rect(FD.cols / 2, FD.rows / 2, FD.cols / 2 - 1, FD.rows / 2 - 1));
	deltaF2 = FD(Rect(FD.cols / 4, FD.rows / 4, FD.cols / 4, FD.rows / 4));
	deltaF3 = FD(Rect(FD.cols / 8, FD.rows / 8, FD.cols / 8, FD.rows / 8));
	E1 = ED(Rect(ED.cols / 2, ED.rows / 2, ED.cols / 2 - 1, ED.rows / 2 - 1));
	E2 = ED(Rect(ED.cols / 4, ED.rows / 4, ED.cols / 4, ED.rows / 4));
	E3 = ED(Rect(ED.cols / 8, ED.rows / 8, ED.cols / 8, ED.rows / 8));

	resize(deltaF2, deltaF2, deltaF1.size());
	resize(deltaF3, deltaF3, deltaF1.size());
	resize(E2, E2, E1.size());
	resize(E3, E3, E1.size());

	deltaF2 = (deltaF2 - deltaF1) / 81;
	deltaF3 = (deltaF3 - deltaF2) / 81;
	exp(-deltaF2, deltaF2);
	exp(-deltaF3, deltaF3);
	C2 = 1 - deltaF2;
	C3 = 1 - deltaF3;

	S = E1 + C2.mul(E2) + C3.mul(E3);
	for (int r = 0; r < S.rows; r++)
	{
		for (int c = 0; c < S.cols; c++)
		{
			for (int n = 0; n < S.channels(); n++)
			{
				//S.ptr<float>(r, c)[n] /= 1e-5;
				float pixel = S.ptr<float>(r, c)[n];

				//<<<<<<< HEAD
				if (pixel > 2e-3 || pixel < 0 || pixel == NAN)	S.ptr<float>(r, c)[n] = 0;

			}
		}
	}
	cout << ((double)cvGetTickCount() - Time) / (cvGetTickFrequency() * 1000) << endl;
	return 0;
}

vector<pair<string, Layer>> objectName = { { "BaofengRegion"	,BaofengRegion },
											{ "SteelRegion"		,SteelRegion},
											{ "PadRegion"		,PadRegion} };

void getDataFromFile()
{
	string rootDir = "D:/FPCB/items/output/";
	stringstream ss;
	stringstream filename;
	APIFPCBInspector inspector;

	Mat objImage = imread(rootDir + "1-L.jpg");
	if (objImage.empty())	return;

	for (int i = 1; i >= 0; i--)
	{
		ss << i;
		string path = rootDir + ss.str() + "/";
		if (_access(path.c_str(), 0) == -1)			break;

		//提取目标掩模
		for (int vo = 0; vo < objectName.size(); vo++)
		{
			string objStr = objectName[vo].first;
			Layer objType = objectName[vo].second;
			if (_access((path + objStr + "/").c_str(), 0) != -1)
			{
				string dirPath = path + objStr + "/";
				string resPath = dirPath + "res.json";
				ifstream jsonFile(resPath);
				json jsonParam;
				jsonFile >> jsonParam;
				for (int j = 0; j < jsonParam.size(); j++)
				{
					PART_REGION partRegion;
					string strParam = jsonParam[j]["imgName"];
					string strType = jsonParam[j]["type"];
					string maskFileName;
					string maskType;

					int nameT = strParam.find_last_of('/') + 1;
					if (nameT == 0)		maskFileName = strParam;
					else				maskFileName = strParam.substr(nameT);

					partRegion.mask = imread(dirPath + maskFileName);
					partRegion.type = objType;
					partRegion.iOffsetX = jsonParam[j]["iOffsetX"];
					partRegion.iOffsetY = jsonParam[j]["iOffsetY"];
				}
			}
		}
		ss.str("");
	}
}
*/

///////////////////////////////////////////////////////////////
//location
//////////////////////////////////////////////////////////////

void parseParm(std::string parmPath, CoarseMatchData_Input& coarseParm)
{
	std::string szContent = "gerber_date_attribute";
	std::string szgerbDPI = "gerbDPI";
	//std::string szmachineName = "machineName";
	//std::string szmaxAngleDeviation = "maxAngleDeviation";
	//std::string szscaleFactor = "scaleFactor";
	std::string szrowMacDist_gerb = "mark_point_row";
	std::string szcolMacDist_gerb = "mark_point_col";

	std::string szPointRow = "point_row";
	std::string szPointCol = "point_col";
	//std::string szszGerbPath = "szGerbPath";

	std::string szDuplicateDeg = "iDuplicateDeg";
	//std::string szimgWidth = "imgWidth";
	//std::string szimgHeight = "imgHeight";
	//std::string sziImgNumber = "iImgNumber";

	std::ifstream ifile(parmPath);
	if (!ifile.is_open())
		return;

	nlohmann::json j;
	ifile >> j;
	//std::cout << /*std::setw(4)*//* <<*/ j << std::endl;



	for (auto it = j.begin(); it != j.end(); it++)
	{
		if (it.key() == szContent)
		{
			auto& contentValue = it.value();

			for (auto itt = contentValue.begin(); itt != contentValue.end(); itt++)
			{
				if (itt.key() == szgerbDPI)
				{
					coarseParm.gerbDPI = itt.value();
				}
				// 		else if (it.key() == szmachineName)
				// 		{
				// 			coarseParm.szMachineName = it.value();
				// 		}
				// 		else if (it.key() == szmaxAngleDeviation)
				// 		{
				// 			coarseParm.maxAngleDeviation = it.value();
				// 		}
				else if (itt.key() == szrowMacDist_gerb)
				{
					auto& arrRowsValue = itt.value();
					std::vector< std::vector<float> > rowsMacRegion_gerb;

					for (auto arrRowsIt = arrRowsValue.begin(); arrRowsIt != arrRowsValue.end(); arrRowsIt++)
					{
						std::vector<float> ptRows;
						auto& singleRowsValue = arrRowsIt.value();
						for (auto singleRowsIt = singleRowsValue.begin(); singleRowsIt != singleRowsValue.end(); singleRowsIt++)
						{
							auto& rowValue = singleRowsIt.value();
							for (auto v = rowValue.begin(); v != rowValue.end(); v++)
							{
								if (v.key() == szPointRow)
								{
									float fRow = v.value();
									ptRows.emplace_back(fRow);
								}
							}

						}

						rowsMacRegion_gerb.emplace_back(ptRows);
					}

					coarseParm.rowsMacRegion_gerb = rowsMacRegion_gerb;
				}
				else if (itt.key() == szcolMacDist_gerb)
				{
					auto& arrColsValue = itt.value();
					std::vector< std::vector<float> > colsMacRegion_gerb;

					for (auto arrColsIt = arrColsValue.begin(); arrColsIt != arrColsValue.end(); arrColsIt++)
					{
						std::vector<float> ptCols;
						auto& singleColsValue = arrColsIt.value();
						for (auto singleColsIt = singleColsValue.begin(); singleColsIt != singleColsValue.end(); singleColsIt++)
						{
							auto& colValue = singleColsIt.value();
							for (auto v = colValue.begin(); v != colValue.end(); v++)
							{
								if (v.key() == szPointCol)
								{
									float fCol = v.value();
									ptCols.emplace_back(fCol);
								}
							}
						}

						colsMacRegion_gerb.emplace_back(ptCols);
					}

					coarseParm.colsMacRegion_gerb = colsMacRegion_gerb;
				}
				// 		else if (it.key() == szscaleFactor)
				// 		{
				// 			coarseParm.scaleFactor = it.value();
				// 		}
				// 		else if (it.key() == szszGerbPath)
				// 		{
				// 			std::string fileName = it.value();
				// 
				// 			int iPos = parmPath.rfind("\\");
				// 			std::string filePath = parmPath.substr(0, iPos + 1) + fileName;
				// 			coarseParm.imgGerb = cv::imread(filePath, 0);
				// 		}
				else if (itt.key() == szDuplicateDeg)
				{
					coarseParm.iDuplicateDeg = itt.value();
				}
				// 		else if (it.key() == szimgWidth)
				// 		{
				// 			coarseParm.iImgWidth = it.value();
				// 		}
				// 		else if (it.key() == szimgHeight)
				// 		{
				// 			coarseParm.iImgHeight = it.value();
				// 		}
				// 		else if (it.key() == sziImgNumber)
				// 		{
				// 			coarseParm.iImgNumber = it.value();
				// 		}
			}


		}
	}

	return;
}

int parseJ(const std::vector<std::string>& vecJsonPath, LayerData& layData)
{
	CParseJ j;
	j.parseJ(vecJsonPath);
	layData = j.m_mapLayItems;
	return 0;
}

#include <thread>

//void run2()
//{
//	APIFPCBInspector m_APIFPCBInspector;
//	CoarseMatchData matchData;
//	bm::pureInterface::CUiDataInteraction_api *pCUiDataInteraction_api;
//	pCUiDataInteraction_api = bm::pureInterface::GetUiDataInteraction();
//
//	std::string mProductName1 = "A";
//	pCUiDataInteraction_api->setProductName(mProductName1);
//
//	Layout_Polarity mLayout_Polarity = mPositive;
//	bmp_type mbmp_type = mBinary;
//	std::string path = pCUiDataInteraction_api->getImageDirectory(mLayout_Polarity, mbmp_type);
//
//	//读取辅助信息数据
//	auxiliaryDataType mauxiliaryData = pCUiDataInteraction_api->readAuxiliaryData();
//	if (mauxiliaryData.positiveAuxiliaryDataAttribute)
//	{
//		auxiliaryDataAttribute bmpositiveauxiliarydata = *mauxiliaryData.positiveAuxiliaryDataAttribute;
//		matchData.colMacDist_gerb = bmpositiveauxiliarydata.colMacDist_gerb;
//		matchData.rowMacDist_gerb = bmpositiveauxiliarydata.rowMacDist_gerb;
//		matchData.gerbDPI = bmpositiveauxiliarydata.gerbDPI;
//	}
//	// 	if (mauxiliaryData.negativeAuxiliaryDataAttribute)
//	// 	{
//	// 		auxiliaryDataAttribute bmnegativeauxiliarydata = *mauxiliaryData.negativeAuxiliaryDataAttribute;
//	// 		matchData.colMacDist_gerb = bmnegativeauxiliarydata.colMacDist_gerb;
//	// 		matchData.rowMacDist_gerb = bmnegativeauxiliarydata.rowMacDist_gerb;
//	// 	}
//	if (mauxiliaryData.commonAuxiliaryDataAttribute)
//	{
//		auxiliaryDataAttribute bmcommonauxiliarydata = *mauxiliaryData.commonAuxiliaryDataAttribute;
//		matchData.colMacDist_gerb = bmcommonauxiliarydata.colMacDist_gerb;
//		matchData.rowMacDist_gerb = bmcommonauxiliarydata.rowMacDist_gerb;
//	}
//
//	LayerDataType mLayerDataType;
//	bm::pureInterface::CUiDataInteraction_api *bmpCUiDataInteraction_api;
//	bmpCUiDataInteraction_api = bm::pureInterface::GetUiDataInteraction();
//	bmpCUiDataInteraction_api->setProductName(mProductName1);
//	//先加载，在获取，顺序不能反
//	list<string> prodlist = bmpCUiDataInteraction_api->getProductList();
//	bmpCUiDataInteraction_api->loadProductData();
//	//pCUiDataInteraction_api->getProductData()返回的是带有三个map的结构体，获取赋值前需先判断map是否为空，为空则获取赋值
//	mLayerDataType = bmpCUiDataInteraction_api->getProductData();
//
//	if (mLayerDataType.positiveLayerData)
//	{
//		matchData.layGerb = mLayerDataType.positiveLayerData;
//	}
//	// 	if (mLayerDataType.negativeLayerData)
//	// 	{
//	// 		matchData.layGerb = mLayerDataType.negativeLayerData;
//	// 	}
//	if (mLayerDataType.commonLayerData)
//	{
//		matchData.layGerb = mLayerDataType.commonLayerData;
//	}
//
//	Mat imgrend = imread(path, 0);
//
//	ImageInfo imgrendS;
//	imgrendS.height = imgrend.rows;
//	imgrendS.width = imgrend.cols;
//	imgrendS.ptr = imgrend.data;    //ptr;
//	imgrendS.step = imgrend.step[0];//imgrendS.width +1;
//	matchData.imgGerb = imgrendS;
//
//	ImageInfo output;
//	output.ptr = new unsigned char[imgrendS.height * imgrendS.width * 3];
//	output.width = matchData.imgGerb.width;
//	output.height = matchData.imgGerb.height;
//	output.step = matchData.imgGerb.width * 3;
//	matchData.iImgHeight = 15000;
//	matchData.iImgWidth = 8192;
//	matchData.iImgNumber = 2;
//
//	int isTop = 0;//正面是0，反面是1
//	m_APIFPCBInspector.api_gerber_location(matchData, output, isTop);
//
//	Mat output_image(output.height, output.width, CV_8UC3, output.ptr, output.step);
//	cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
//	imwrite("1.bmp", output_image);
//
//}


void run()
{

    ConfigParam topParam, backParam;

	APIFPCBInspector fpcb;
    string mzName = "S2";
    //CalibData cdata;
    //fpcb.api_getCalibData("S2",cdata);
	Mat lImage, rImage, image3;
	std::string path =  "D:/other/10404-0927/bottom";

	CoarseMatchData_Input posData, negData;
	string bmDir;
	CUtility::getExePath(bmDir);
	//parseParm(bmDir + "/parm/DZ10259/Positive/markmsg.json", posData);
	//parseParm(bmDir + "/parm/DZ10259/Negative/markmsg.json", negData);

	parseParm(bmDir + "/parm/DZ10404/Positive/markmsg.json", posData);
	parseParm(bmDir + "/parm/DZ10404/Negative/markmsg.json", negData);

	std::vector<std::string> posVecJsonFilePath, negVecJsonFilePath;
	//CUtility::getFiles((bmDir + "/parm/DZ10259/Positive/").c_str(), ".bm", posVecJsonFilePath);
	//CUtility::getFiles((bmDir + "/parm/DZ10259/Negative/").c_str(), ".bm", negVecJsonFilePath);

	CUtility::getFiles((bmDir + "/parm/DZ10404/Positive/").c_str(), ".bm", posVecJsonFilePath);
	CUtility::getFiles((bmDir + "/parm/DZ10404/Negative/").c_str(), ".bm", negVecJsonFilePath);

	LayerData posLayData, negLayData;
	parseJ(posVecJsonFilePath, posLayData);
	parseJ(negVecJsonFilePath, negLayData);

	posData.layGerb = posLayData;
	negData.layGerb = negLayData;

	std::cout << "加载完毕" << endl;

	cv::Mat imgRendergraphPos;
	CoarseMatchData _posData;
	{
		//imgRendergraphPos = cv::imread(bmDir + "/parm/DZ10259/Positive/rendergraph.bmp",0);
		imgRendergraphPos = cv::imread(bmDir + "/parm/DZ10404/Positive/rendergraph.bmp", 0);
		// 
		_posData.imgGerb.ptr = imgRendergraphPos.data;
		_posData.imgGerb.width = imgRendergraphPos.cols;
		_posData.imgGerb.height = imgRendergraphPos.rows;
		_posData.imgGerb.step = imgRendergraphPos.step;

		_posData.layGerb = &posData.layGerb;

		//废弃
		_posData.rowMacDist_gerb = posData.rowsMacRegion_gerb;
		_posData.colMacDist_gerb = posData.colsMacRegion_gerb;

		_posData.gerbDPI = 1200;
		//_posData.pixelRatio = posData.pixelRatio;
		//_data.lineSpeed = data.lineSpeed;
		//_data.lineFreq = data.lineFreq;
		_posData.szMachineName = "D1A";
		_posData.maxAngleDeviation = 5.0;
		_posData.scaleFactor = 0.125;
		_posData.iDuplicateDeg = 180;

		_posData.iImgWidth = 8192;
		_posData.iImgHeight = 26500;
		_posData.iImgNumber = 3;
	}
	Mat _temp(_posData.imgGerb.height, _posData.imgGerb.width, CV_8UC1, _posData.imgGerb.ptr);

	cv::Mat imgRendergraphNegpos;
	CoarseMatchData _negData;
	{
		//imgRendergraphNegpos = cv::imread(bmDir + "/parm/DZ10259/Negative/rendergraph.bmp",0);
		imgRendergraphNegpos = cv::imread(bmDir + "/parm/DZ10404/Negative/rendergraph.bmp", 0);
		//替换
		_negData.imgGerb.ptr = imgRendergraphNegpos.data;
		_negData.imgGerb.width = imgRendergraphNegpos.cols;
		_negData.imgGerb.height = imgRendergraphNegpos.rows;
		_negData.imgGerb.step = imgRendergraphNegpos.step;

		
		_negData.layGerb = &negData.layGerb;
		_negData.rowMacDist_gerb = negData.rowsMacRegion_gerb;
		_negData.colMacDist_gerb = negData.colsMacRegion_gerb;

		_negData.gerbDPI = 1200;
		_negData.szMachineName = "D1A";
		//_negData.pixelRatio = negData.pixelRatio;
		//_data.lineSpeed = data.lineSpeed;
		//_data.lineFreq = data.lineFreq;
		_negData.maxAngleDeviation = 5.0;
		_negData.scaleFactor = 0.125;
		_negData.iDuplicateDeg = 180;
		_negData.iImgWidth = 8192;
		_negData.iImgHeight = 26500;
		_negData.iImgNumber = 3;
	}

	std::vector<std::vector<std::string>> frontPath, backPath;
	paresPath(path, frontPath, backPath,3);

// 	ImageInfo output;
	OutputInfo *out = new OutputInfo();
// 	float scale = 1;
// 	output.ptr = (unsigned char*)malloc(scale * _posData.iImgWidth * scale * _posData.iImgHeight * 3 * sizeof(unsigned char));
// 	output.width = _posData.iImgWidth * scale;
// 	output.height = _posData.iImgHeight * scale;
// 	output.step = output.width * 3;
// 	output.nLabel = 1;


	//goto back;
	fpcb.api_init(_posData, 1);

    vector<DefectRoi> defs;

	/*for (;;)*/
	{
		for (int i = 0; i < /*frontPath.size()*/1; i++)
		{
			//lImage = imread(frontPath[i][0]);
			//rImage = imread(frontPath[i][1]);
			//
			//lImage = imread(path + "/alg_20190812101446_1_0_2_0_f.jpg");
			//rImage = imread(path + "/alg_20190812101446_1_1_2_1_f.jpg");

			lImage = imread("D:/other/10404-0927/top/alg_20190927135627_1_27_3_0_f.jpg");
			rImage = imread("D:/other/10404-0927/top/alg_20190927135627_1_28_3_1_f.jpg");
			image3 = imread("D:/other/10404-0927/top/alg_20190927135627_1_30_3_2_f.jpg");

			cv::cvtColor(lImage,lImage,cv::ColorConversionCodes::COLOR_BGR2RGB);
			cv::cvtColor(rImage, rImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
			cv::cvtColor(image3, image3, cv::ColorConversionCodes::COLOR_BGR2RGB);
 
			ImageInfo imageVec1, imageVec2, imageVec3;
			imageVec1 = { lImage.data,lImage.cols, lImage.rows, lImage.step, 0 };
			imageVec2 = { rImage.data,rImage.cols, rImage.rows, rImage.step, 1 };
			imageVec3 = { image3.data,image3.cols, image3.rows, image3.step, 2 };
			
            //std::vector<ImageInfo> ilr = { imageVec1,imageVec2 };

			//cv::Mat gerberlocation(output.height, output.width, CV_8UC3, output.ptr, output.step);

			//fpcb.api_gerber_location(output, 1);
			
			//cvtColor(mout, mout, cv::COLOR_BGR2RGB);
			fpcb.api_loadParam("../reverse/parameter/");

			//thread t1(&APIFPCBInspector::api_learn, fpcb, imageVec1, 1);
			//thread t2(&APIFPCBInspector::api_learn, fpcb, imageVec2, 1);
			//t1.join();
			//t2.join();

			/*ImageInfo output;
			output.height = _posData.iImgHeight;
			output.width = _posData.iImgWidth;
			output.step = output.width * 3;
			output.ptr = (uchar*)malloc(output.step*output.height * sizeof(uchar));

			fpcb.api_gerber_location(_posData, output, 1, 1);

			cv::Mat imgShow(output.height, output.width, CV_8UC3, output.ptr);*/

			if (i == 0)
			{
				fpcb.api_learn(imageVec1, 1);
				fpcb.api_learn(imageVec2, 1);
				fpcb.api_learn(imageVec3, 1);
                fpcb.api_get_configParam(topParam, 1);

                ofstream outTop("topParam.dat", ios::binary);

                outTop.write((const char*)&topParam, sizeof(ConfigParam));

                outTop.close();
                
				//fpcb.api_save_configparam(&cparam);
			}

            ifstream inTop("topParam.dat",ios::binary);

            inTop.read((char*)&topParam, sizeof(ConfigParam));

            fpcb.api_set_configParam(topParam, 1);

			out->scale = 0.5;
			out->image.height = _posData.iImgHeight;
			out->image.width = _posData.iImgWidth;

			out->image.ptr = (unsigned char*)malloc(sizeof(unsigned char) * _posData.iImgWidth * _posData.iImgHeight * 3);

			fpcb.api_load_space(out);

            //fpcb.api_processInterface(imageVec1, imageVec1.nLabel, 1, defs);

			fpcb.api_process(imageVec1, 1);
			//fpcb.api_process(imageVec2, 1);
			//fpcb.api_process(imageVec3, 1);
		}

	}
	
	//back:
	
	fpcb.api_init(_negData, 0);

	for (int i = 0; i < 1/*backPath.size()*/; i++)
	{
		//lImage = imread(backPath[i][0]);// backPath[i][0]);
		//rImage = imread(backPath[i][1]);

																																																																																																																																																																																																																																																																																																																																								
		/*lImage = imread("E:\\FPCB\\10404\\alg_20190821092739__0_3_0_r.jpg");
		rImage = imread("E:\\FPCB\\10404\\alg_20190821092739__1_3_1_r.jpg");
		image3 = imread("E:\\FPCB\\10404\\alg_20190821092739__2_3_2_r.jpg");*/


		lImage = imread(path + "/alg_20190927135738_1_33_3_0_r.jpg");
		rImage = imread(path + "/alg_20190927135738_1_34_3_1_r.jpg");
		image3 = imread(path + "/alg_20190927135738_1_35_3_2_r.jpg");

		cv::cvtColor(lImage, lImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
		cv::cvtColor(rImage, rImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
		cv::cvtColor(image3, image3, cv::ColorConversionCodes::COLOR_BGR2RGB);

		ImageInfo imageVec1, imageVec2, imageVec3;
		imageVec1 = { lImage.data,lImage.cols, lImage.rows, lImage.step, 0 };
		imageVec2 = { rImage.data,rImage.cols, rImage.rows, rImage.step, 1 };
		imageVec3 = { image3.data,image3.cols, image3.rows, image3.step, 2 };

		fpcb.api_loadParam("../reverse/parameter/");

		//cv::Mat gerberlocation(output.height, output.width, CV_8UC3, output.ptr, output.step);

		//fpcb.api_gerber_location(output, 0);

		if (i == 0)
		{
			fpcb.api_learn(imageVec1, 0);
			fpcb.api_learn(imageVec2, 0);
			fpcb.api_learn(imageVec3, 0);

            fpcb.api_get_configParam(backParam, 0);

            ofstream outTop("backParam.dat", ios::binary);

            outTop.write((const char*)&backParam, sizeof(ConfigParam));

            outTop.close();
		}

// 		fpcb.api_location(&output, scale, 2);
//
// 		cv::imwrite(std::to_string(i) + "_pos.jpg", mout);
//

        ifstream inBack("backParam.dat", ios::binary);

        inBack.read((char*)&backParam, sizeof(ConfigParam));

        fpcb.api_set_configParam(backParam, 1);

		out->scale = 0.5;
		out->image.height = _negData.iImgHeight;
		out->image.width = _negData.iImgWidth;
		out->image.ptr = (unsigned char*)malloc(sizeof(unsigned char) * _negData.iImgWidth * _negData.iImgHeight * 3);
		fpcb.api_load_space(out);
		
		fpcb.api_process(imageVec1, 0);
		fpcb.api_process(imageVec2, 0);
		fpcb.api_process(imageVec3, 0);


		//thread t1(&APIFPCBInspector::api_process, fpcb, imageVec1, 0);
		//thread t2(&APIFPCBInspector::api_process, fpcb, imageVec2, 0);
		//t1.join();
		//t2.join();
	}
}

void run10259()
{
	APIFPCBInspector fpcb;
	string mzName = "S2";
	//CalibData cdata;
	//fpcb.api_getCalibData("S2",cdata);
	Mat lImage, rImage, image3;
	std::string path = "E:/FPCB/2019-08-12";

	CoarseMatchData_Input posData, negData;
	string bmDir;
	CUtility::getExePath(bmDir);
	//parseParm(bmDir + "/parm/DZ10259/Positive/markmsg.json", posData);
	//parseParm(bmDir + "/parm/DZ10259/Negative/markmsg.json", negData);

	parseParm(bmDir + "/parm/DZ10259/Positive/markmsg.json", posData);
	parseParm(bmDir + "/parm/DZ10259/Negative/markmsg.json", negData);

	std::vector<std::string> posVecJsonFilePath, negVecJsonFilePath;
	/*CUtility::getFiles((bmDir + "/parm/DZ10259/Positive/").c_str(), ".bm", posVecJsonFilePath);
	CUtility::getFiles((bmDir + "/parm/DZ10259/Negative/").c_str(), ".bm", negVecJsonFilePath);*/

	CUtility::getFiles((bmDir + "/parm/DZ10259/Positive/").c_str(), ".bm", posVecJsonFilePath);
	CUtility::getFiles((bmDir + "/parm/DZ10259/Negative/").c_str(), ".bm", negVecJsonFilePath);

	LayerData posLayData, negLayData;
	parseJ(posVecJsonFilePath, posLayData);
	parseJ(negVecJsonFilePath, negLayData);

	posData.layGerb = posLayData;
	negData.layGerb = negLayData;

	std::cout << "加载完毕" << endl;

	cv::Mat imgRendergraphPos;
	CoarseMatchData _posData;
	{
		//imgRendergraphPos = cv::imread(bmDir + "/parm/DZ10259/Positive/rendergraph.bmp",0);
		imgRendergraphPos = cv::imread(bmDir + "/parm/DZ10259/Positive/rendergraph.bmp", 0);
		// 
		_posData.imgGerb.ptr = imgRendergraphPos.data;
		_posData.imgGerb.width = imgRendergraphPos.cols;
		_posData.imgGerb.height = imgRendergraphPos.rows;
		_posData.imgGerb.step = imgRendergraphPos.step;

		_posData.layGerb = &posData.layGerb;

		//废弃
		_posData.rowMacDist_gerb = posData.rowsMacRegion_gerb;
		_posData.colMacDist_gerb = posData.colsMacRegion_gerb;

		_posData.gerbDPI = 1200;
		//_posData.pixelRatio = posData.pixelRatio;
		//_data.lineSpeed = data.lineSpeed;
		//_data.lineFreq = data.lineFreq;
		_posData.szMachineName = "D1A";
		_posData.maxAngleDeviation = 5.0;
		_posData.scaleFactor = 0.125;
		_posData.iDuplicateDeg = 180;


		_posData.iImgWidth = 8192;
		_posData.iImgHeight = 15000;
		_posData.iImgNumber = 2;
	}
	

	cv::Mat imgRendergraphNegpos;
	CoarseMatchData _negData;
	{
		//imgRendergraphNegpos = cv::imread(bmDir + "/parm/DZ10259/Negative/rendergraph.bmp",0);
		imgRendergraphNegpos = cv::imread(bmDir + "/parm/DZ10259/Negative/rendergraph.bmp", 0);
		//替换
		_negData.imgGerb.ptr = imgRendergraphNegpos.data;
		_negData.imgGerb.width = imgRendergraphNegpos.cols;
		_negData.imgGerb.height = imgRendergraphNegpos.rows;
		_negData.imgGerb.step = imgRendergraphNegpos.step;


		_negData.layGerb = &negData.layGerb;
		_negData.rowMacDist_gerb = negData.rowsMacRegion_gerb;
		_negData.colMacDist_gerb = negData.colsMacRegion_gerb;

		_negData.gerbDPI = 1200;
		_negData.szMachineName = "D1A";
		//_negData.pixelRatio = negData.pixelRatio;
		//_data.lineSpeed = data.lineSpeed;
		//_data.lineFreq = data.lineFreq;
		_negData.maxAngleDeviation = 5.0;
		_negData.scaleFactor = 0.125;
		_negData.iDuplicateDeg = 180;
		_negData.iImgWidth = 8192;
		_negData.iImgHeight = 15000;
		_negData.iImgNumber = 2;
	}

	std::vector<std::vector<std::string>> frontPath, backPath;
	paresPath(path, frontPath, backPath, 2);


	OutputInfo *out = new OutputInfo();
	float scale = out->scale = 1;
	out->image.ptr = (unsigned char*)malloc(scale * _posData.iImgWidth * scale * _posData.iImgHeight * 3 * sizeof(unsigned char)*_posData.iImgNumber);
	out->image.width = _posData.iImgWidth * scale*_posData.iImgNumber;
	out->image.height = _posData.iImgHeight * scale;
	out->image.step = out->image.width * 3;
	out->image.nLabel = 1;

	ConfigParam cparam;

	//goto back;
	fpcb.api_init(_posData, 1);

	/*for (;;)*/
	{
		for (int i = 0; i < 10/*frontPath.size()*/; i++)
		{
			//lImage = imread(frontPath[i][0]);
			//rImage = imread(frontPath[i][1]);

			/*lImage = imread("E:\\FPCB\\NG\\alg_20190814090228_3_8_2_0_f.jpg");
			rImage = imread("E:\\FPCB\\NG\\alg_20190814090228_3_9_2_1_f.jpg");*/


			/*lImage = imread("E:\\FPCB\\10404\\alg_20190820143206_8_18_3_0_f.jpg");
			rImage = imread("E:\\FPCB\\10404\\alg_20190820143206_8_19_3_1_f.jpg");
			image3 = imread("E:\\FPCB\\10404\\alg_20190820143206_8_20_3_2_f.jpg");*/

			lImage = imread("E:\\FPCB\\2019-09-24\\alg_20190924094208_4_38_2_0_f.jpg");
			rImage = imread("E:\\FPCB\\2019-09-24\\alg_20190924094208_4_39_2_1_f.jpg");
			//image3 = imread("E:\\FPCB\\新光源\\2019-09-121\\alg_20190912151305__62_3_2_f.jpg");

			cv::cvtColor(lImage, lImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
			cv::cvtColor(rImage, rImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
			//cv::cvtColor(image3, image3, cv::ColorConversionCodes::COLOR_BGR2RGB);

			ImageInfo imageVec1, imageVec2, imageVec3;
			imageVec1 = { lImage.data,lImage.cols, lImage.rows, lImage.step, 0 };
			imageVec2 = { rImage.data,rImage.cols, rImage.rows, rImage.step, 1 };
			//imageVec3 = { image3.data,image3.cols, image3.rows, image3.step, 2 };
			//std::vector<ImageInfo> ilr = { imageVec1,imageVec2 };

			//cv::Mat gerberlocation(output.height, output.width, CV_8UC3, output.ptr, output.step);

			//fpcb.api_gerber_location(output, 1);

			//cvtColor(mout, mout, cv::COLOR_BGR2RGB);
			fpcb.api_loadParam("../reverse/parameter/10259/");

			//thread t1(&APIFPCBInspector::api_learn, fpcb, imageVec1, 1);
			//thread t2(&APIFPCBInspector::api_learn, fpcb, imageVec2, 1);
			//t1.join();
			//t2.join();

			/*ImageInfo output;
			output.height = _posData.iImgHeight;
			output.width = _posData.iImgWidth;
			output.step = output.width * 3;
			output.ptr = (uchar*)malloc(output.step*output.height * sizeof(uchar));

			fpcb.api_gerber_location(_posData, output, 1, 1);

			cv::Mat imgShow(output.height, output.width, CV_8UC3, output.ptr);*/

			/*if (i == 0)
			{
				fpcb.api_learn(imageVec1, 1);
				fpcb.api_learn(imageVec2, 1);
				fpcb.api_learn(imageVec3, 1);
				fpcb.api_save_configparam(&cparam);
			}*/

			fpcb.api_load_space(out);


			fpcb.api_process(imageVec1, 1);
			fpcb.api_process(imageVec2, 1);
			//fpcb.api_process(imageVec3, 1);
		}

	}

	//back:

	fpcb.api_init(_negData, 0);

	for (int i = 0; i < 1/*backPath.size()*/; i++)
	{
		//lImage = imread(backPath[i][0]);// backPath[i][0]);
		//rImage = imread(backPath[i][1]);


		/*lImage = imread("E:\\FPCB\\10404\\alg_20190821092739__0_3_0_r.jpg");
		rImage = imread("E:\\FPCB\\10404\\alg_20190821092739__1_3_1_r.jpg");
		image3 = imread("E:\\FPCB\\10404\\alg_20190821092739__2_3_2_r.jpg");*/


		/*lImage = imread("E:\\FPCB\\10404\\alg_20190826144014_1_0_3_0_r.jpg");
		rImage = imread("E:\\FPCB\\10404\\alg_20190826144014_1_1_3_1_r.jpg");
		image3 = imread("E:\\FPCB\\10404\\alg_20190826144014_1_2_3_2_r.jpg");*/

		lImage = imread("E:\\FPCB\\2019-09-19-test\\alg_20190919174037_21_194_2_0_r.jpg");
		rImage = imread("E:\\FPCB\\2019-09-19-test\\alg_20190919174037_21_195_2_1_r.jpg");
		//image3 = imread("E:\\FPCB\\新光源\\2019-09-121\\alg_20190912104920_10_50_3_2_r.jpg");

		cv::cvtColor(lImage, lImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
		cv::cvtColor(rImage, rImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
		//cv::cvtColor(image3, image3, cv::ColorConversionCodes::COLOR_BGR2RGB);

		ImageInfo imageVec1, imageVec2, imageVec3;
		imageVec1 = { lImage.data,lImage.cols, lImage.rows, lImage.step, 0 };
		imageVec2 = { rImage.data,rImage.cols, rImage.rows, rImage.step, 1 };
		//imageVec3 = { image3.data,image3.cols, image3.rows, image3.step, 2 };


		fpcb.api_loadParam("../reverse/parameter/10259/");

		//cv::Mat gerberlocation(output.height, output.width, CV_8UC3, output.ptr, output.step);

		//fpcb.api_gerber_location(output, 0);
		/*if (i == 0)
		{
			fpcb.api_learn(imageVec1, 0);
			fpcb.api_learn(imageVec2, 0);
			fpcb.api_learn(imageVec3, 0);

			fpcb.api_save_configparam(&cparam);
		}*/
		// 		fpcb.api_location(&output, scale, 2);
		//
		//
		// 		cv::imwrite(std::to_string(i) + "_pos.jpg", mout);
		fpcb.api_load_space(out);

		fpcb.api_process(imageVec1, 0);
		fpcb.api_process(imageVec2, 0);
		//fpcb.api_process(imageVec3, 0);


		//thread t1(&APIFPCBInspector::api_process, fpcb, imageVec1, 0);
		//thread t2(&APIFPCBInspector::api_process, fpcb, imageVec2, 0);
		//t1.join();
		//t2.join();
	}
}


void run10180()
{
	APIFPCBInspector fpcb;
	string mzName = "S2";
	//CalibData cdata;
	//fpcb.api_getCalibData("S2",cdata);
	Mat lImage, rImage, image3;
	std::string path = "E:\FPCB\2019-08-12";

	CoarseMatchData_Input posData, negData;
	string bmDir;
	CUtility::getExePath(bmDir);
	//parseParm(bmDir + "/parm/DZ10259/Positive/markmsg.json", posData);
	//parseParm(bmDir + "/parm/DZ10259/Negative/markmsg.json", negData);

	parseParm(bmDir + "/parm/DZ10404C/Positive/markmsg.json", posData);
	parseParm(bmDir + "/parm/DZ10404C/Negative/markmsg.json", negData);

	std::vector<std::string> posVecJsonFilePath, negVecJsonFilePath;
	/*CUtility::getFiles((bmDir + "/parm/DZ10259/Positive/").c_str(), ".bm", posVecJsonFilePath);
	CUtility::getFiles((bmDir + "/parm/DZ10259/Negative/").c_str(), ".bm", negVecJsonFilePath);*/

	CUtility::getFiles((bmDir + "/parm/DZ10404C/Positive/").c_str(), ".bm", posVecJsonFilePath);
	CUtility::getFiles((bmDir + "/parm/DZ10404C/Negative/").c_str(), ".bm", negVecJsonFilePath);

	LayerData posLayData, negLayData;
	parseJ(posVecJsonFilePath, posLayData);
	parseJ(negVecJsonFilePath, negLayData);

	posData.layGerb = posLayData;
	negData.layGerb = negLayData;

	std::cout << "加载完毕" << endl;

	cv::Mat imgRendergraphPos;
	CoarseMatchData _posData;
	{
		//imgRendergraphPos = cv::imread(bmDir + "/parm/DZ10259/Positive/rendergraph.bmp",0);
		imgRendergraphPos = cv::imread(bmDir + "/parm/DZ10404C/Positive/rendergraph.bmp", 0);
		// 
		_posData.imgGerb.ptr = imgRendergraphPos.data;
		_posData.imgGerb.width = imgRendergraphPos.cols;
		_posData.imgGerb.height = imgRendergraphPos.rows;
		_posData.imgGerb.step = imgRendergraphPos.step;

		_posData.layGerb = &posData.layGerb;

		//废弃
		_posData.rowMacDist_gerb = posData.rowsMacRegion_gerb;
		_posData.colMacDist_gerb = posData.colsMacRegion_gerb;

		_posData.gerbDPI = 1200;
		//_posData.pixelRatio = posData.pixelRatio;
		//_data.lineSpeed = data.lineSpeed;
		//_data.lineFreq = data.lineFreq;
		_posData.szMachineName = "D1A";
		_posData.maxAngleDeviation = 5.0;
		_posData.scaleFactor = 0.125;
		_posData.iDuplicateDeg = 180;


		_posData.iImgWidth = 8192;
		_posData.iImgHeight = 15000;
		_posData.iImgNumber = 2;
	}


	cv::Mat imgRendergraphNegpos;
	CoarseMatchData _negData;
	{
		//imgRendergraphNegpos = cv::imread(bmDir + "/parm/DZ10259/Negative/rendergraph.bmp",0);
		imgRendergraphNegpos = cv::imread(bmDir + "/parm/DZ10404C/Negative/rendergraph.bmp", 0);
		//替换
		_negData.imgGerb.ptr = imgRendergraphNegpos.data;
		_negData.imgGerb.width = imgRendergraphNegpos.cols;
		_negData.imgGerb.height = imgRendergraphNegpos.rows;
		_negData.imgGerb.step = imgRendergraphNegpos.step;


		_negData.layGerb = &negData.layGerb;
		_negData.rowMacDist_gerb = negData.rowsMacRegion_gerb;
		_negData.colMacDist_gerb = negData.colsMacRegion_gerb;

		_negData.gerbDPI = 1200;
		_negData.szMachineName = "D1A";
		//_negData.pixelRatio = negData.pixelRatio;
		//_data.lineSpeed = data.lineSpeed;
		//_data.lineFreq = data.lineFreq;
		_negData.maxAngleDeviation = 5.0;
		_negData.scaleFactor = 0.125;
		_negData.iDuplicateDeg = 180;
		_negData.iImgWidth = 8192;
		_negData.iImgHeight = 15000;
		_negData.iImgNumber = 2;
	}

	std::vector<std::vector<std::string>> frontPath, backPath;
	paresPath(path, frontPath, backPath, 2);


	OutputInfo *out = new OutputInfo();
	float scale = out->scale = 1;
	out->image.ptr = (unsigned char*)malloc(scale * _posData.iImgWidth * scale * _posData.iImgHeight * 3 * sizeof(unsigned char)*_posData.iImgNumber);
	out->image.width = _posData.iImgWidth * scale*_posData.iImgNumber;
	out->image.height = _posData.iImgHeight * scale;
	out->image.step = out->image.width * 3;
	out->image.nLabel = 1;

	ConfigParam cparam;

	//goto back;
	fpcb.api_init(_posData, 1);

	/*for (;;)*/
	{
		for (int i = 0; i < 10/*frontPath.size()*/; i++)
		{
			//lImage = imread(frontPath[i][0]);
			//rImage = imread(frontPath[i][1]);

			/*lImage = imread("E:\\FPCB\\NG\\alg_20190814090228_3_8_2_0_f.jpg");
			rImage = imread("E:\\FPCB\\NG\\alg_20190814090228_3_9_2_1_f.jpg");*/


			/*lImage = imread("E:\\FPCB\\10404\\alg_20190820143206_8_18_3_0_f.jpg");
			rImage = imread("E:\\FPCB\\10404\\alg_20190820143206_8_19_3_1_f.jpg");
			image3 = imread("E:\\FPCB\\10404\\alg_20190820143206_8_20_3_2_f.jpg");*/

			lImage = imread("E:\\FPCB\\2019-09-24\\2019-09-24\\alg_20190924104446_2_40_2_0_f.jpg");
			rImage = imread("E:\\FPCB\\2019-09-24\\2019-09-24\\alg_20190924104446_2_41_2_1_f.jpg");
			//image3 = imread("E:\\FPCB\\新光源\\2019-09-121\\alg_20190912151305__62_3_2_f.jpg");

			cv::cvtColor(lImage, lImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
			cv::cvtColor(rImage, rImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
			//cv::cvtColor(image3, image3, cv::ColorConversionCodes::COLOR_BGR2RGB);

			ImageInfo imageVec1, imageVec2, imageVec3;
			imageVec1 = { lImage.data,lImage.cols, lImage.rows, lImage.step, 0 };
			imageVec2 = { rImage.data,rImage.cols, rImage.rows, rImage.step, 1 };
			//imageVec3 = { image3.data,image3.cols, image3.rows, image3.step, 2 };
			//std::vector<ImageInfo> ilr = { imageVec1,imageVec2 };

			//cv::Mat gerberlocation(output.height, output.width, CV_8UC3, output.ptr, output.step);

			//fpcb.api_gerber_location(output, 1);

			//cvtColor(mout, mout, cv::COLOR_BGR2RGB);
			fpcb.api_loadParam("../reverse/parameter/10180_1");

			//thread t1(&APIFPCBInspector::api_learn, fpcb, imageVec1, 1);
			//thread t2(&APIFPCBInspector::api_learn, fpcb, imageVec2, 1);
			//t1.join();
			//t2.join();

			/*ImageInfo output;
			output.height = _posData.iImgHeight;
			output.width = _posData.iImgWidth;
			output.step = output.width * 3;
			output.ptr = (uchar*)malloc(output.step*output.height * sizeof(uchar));

			fpcb.api_gerber_location(_posData, output, 1, 1);

			cv::Mat imgShow(output.height, output.width, CV_8UC3, output.ptr);*/

			if (i == 0)
			{
				/*fpcb.api_learn(imageVec1, 1);
				fpcb.api_learn(imageVec2, 1);
				fpcb.api_learn(imageVec3, 1);
				fpcb.api_save_configparam(&cparam);
				*/
			}

			fpcb.api_load_space(out);


			fpcb.api_process(imageVec1, 1);
			fpcb.api_process(imageVec2, 1);
			//fpcb.api_process(imageVec3, 1);
		}

	}

	//back:

	fpcb.api_init(_negData, 0);

	for (int i = 0; i < 1/*backPath.size()*/; i++)
	{
		//lImage = imread(backPath[i][0]);// backPath[i][0]);
		//rImage = imread(backPath[i][1]);


		/*lImage = imread("E:\\FPCB\\10404\\alg_20190821092739__0_3_0_r.jpg");
		rImage = imread("E:\\FPCB\\10404\\alg_20190821092739__1_3_1_r.jpg");
		image3 = imread("E:\\FPCB\\10404\\alg_20190821092739__2_3_2_r.jpg");*/


		/*lImage = imread("E:\\FPCB\\10404\\alg_20190826144014_1_0_3_0_r.jpg");
		rImage = imread("E:\\FPCB\\10404\\alg_20190826144014_1_1_3_1_r.jpg");
		image3 = imread("E:\\FPCB\\10404\\alg_20190826144014_1_2_3_2_r.jpg");*/

		lImage = imread("E:\\FPCB\\2019-09-19-test\\alg_20190919174037_21_194_2_0_r.jpg");
		rImage = imread("E:\\FPCB\\2019-09-19-test\\alg_20190919174037_21_195_2_1_r.jpg");
		//image3 = imread("E:\\FPCB\\新光源\\2019-09-121\\alg_20190912104920_10_50_3_2_r.jpg");

		cv::cvtColor(lImage, lImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
		cv::cvtColor(rImage, rImage, cv::ColorConversionCodes::COLOR_BGR2RGB);
		//cv::cvtColor(image3, image3, cv::ColorConversionCodes::COLOR_BGR2RGB);

		ImageInfo imageVec1, imageVec2, imageVec3;
		imageVec1 = { lImage.data,lImage.cols, lImage.rows, lImage.step, 0 };
		imageVec2 = { rImage.data,rImage.cols, rImage.rows, rImage.step, 1 };
		//imageVec3 = { image3.data,image3.cols, image3.rows, image3.step, 2 };


		fpcb.api_loadParam("../reverse/parameter/10180");

		//cv::Mat gerberlocation(output.height, output.width, CV_8UC3, output.ptr, output.step);

		//fpcb.api_gerber_location(output, 0);
		/*if (i == 0)
		{
		fpcb.api_learn(imageVec1, 0);
		fpcb.api_learn(imageVec2, 0);
		fpcb.api_learn(imageVec3, 0);

		fpcb.api_save_configparam(&cparam);
		}*/
		// 		fpcb.api_location(&output, scale, 2);
		//
		//
		// 		cv::imwrite(std::to_string(i) + "_pos.jpg", mout);
		fpcb.api_load_space(out);

		fpcb.api_process(imageVec1, 0);
		fpcb.api_process(imageVec2, 0);
		//fpcb.api_process(imageVec3, 0);


		//thread t1(&APIFPCBInspector::api_process, fpcb, imageVec1, 0);
		//thread t2(&APIFPCBInspector::api_process, fpcb, imageVec2, 0);
		//t1.join();
		//t2.join();
	}
}

void checkModel(int isTop, std::vector<std::string> imgPath)
{
	for (int i = 0; i < imgPath.size(); i++)
	{
		cv::Mat img = cv::imread(imgPath[i]);
		setTestImage(img, i, isTop);

		std::vector<PolyLay> lays;
		getTestGerbCoarse(lays, isTop);

		if (lays.size() == 0)
			return;


		std::vector<bm::base::bmShapePoint> gerbLTmouses, gerbRBmouse, gerbLToffset, gerbRBoffset;
		if (i == 0)		//细窄线
		{
			gerbLTmouses.emplace_back(bm::base::bmShapePoint(1930, 2242));
			gerbRBmouse.emplace_back(bm::base::bmShapePoint(1952, 5124));
			gerbLToffset.emplace_back(bm::base::bmShapePoint(10, -9));
			gerbRBoffset.emplace_back(bm::base::bmShapePoint(3, 7));
		}
		else if (i == 1)	//宽线
		{
			gerbLTmouses.emplace_back(bm::base::bmShapePoint(6388, 1638));
			gerbRBmouse.emplace_back(bm::base::bmShapePoint(5602, 2188));
			gerbLToffset.emplace_back(bm::base::bmShapePoint(-2, -11));
			gerbRBoffset.emplace_back(bm::base::bmShapePoint(-5, -9));
		}

		setTestPosition(gerbLTmouses, gerbRBmouse, gerbLToffset, gerbRBoffset, isTop);
		getTestGerbModify(lays, isTop);

		doTest(isTop);
	}
	return;
}


void run_getLines_10235()
{
	string bmDir;
	CUtility::getExePath(bmDir);

	CoarseMatchData_Input posData, negData;
	parseParm(bmDir + "/parm/DZ10235/0/parm.json", posData);

	std::vector<std::string> posVecJsonFilePath;
	CUtility::getFiles((bmDir + "/parm/DZ10235/0/").c_str(), ".bm", posVecJsonFilePath);
	
	LayerData posLayData, negLayData;
	parseJ(posVecJsonFilePath, posLayData);

	posData.layGerb = posLayData;
	std::cout << "加载完毕" << endl;

	string mzName = "S2";
	CoarseMatchData _posData; //粗匹配
	{
		_posData.imgGerb.ptr = posData.imgGerb.data;
		_posData.imgGerb.width = posData.imgGerb.cols;
		_posData.imgGerb.height = posData.imgGerb.rows;
		_posData.imgGerb.step = posData.imgGerb.step;
		_posData.layGerb = &posData.layGerb;
		_posData.rowMacDist_gerb = posData.rowsMacRegion_gerb;
		_posData.colMacDist_gerb = posData.colsMacRegion_gerb;
		_posData.gerbDPI = posData.gerbDPI;
		//_posData.pixelRatio = posData.pixelRatio;
		//_data.lineSpeed = data.lineSpeed;
		//_data.lineFreq = data.lineFreq;
		_posData.szMachineName = mzName;
		_posData.maxAngleDeviation = posData.maxAngleDeviation;
		_posData.scaleFactor = posData.scaleFactor;
		_posData.iDuplicateDeg = posData.iDuplicateDeg;
		_posData.iImgWidth = posData.iImgWidth;
		_posData.iImgHeight = posData.iImgHeight;
		_posData.iImgNumber = posData.iImgNumber;
	}

	cv::Mat imgL = cv::imread("E:\\FPCB\\linesregion\\10235\\test_20190515145922_x_0_3_0_f.jpg");
	cv::Mat imgM = cv::imread("E:\\FPCB\\linesregion\\10235\\test_20190515145922_x_1_3_1_f.jpg");
	cv::Mat imgR = cv::imread("E:\\FPCB\\linesregion\\10235\\test_20190515145922_x_2_3_2_f.jpg");

	APIFPCBInspector fpcb;
	fpcb.api_init(_posData, 1);


	std::vector<std::string> imgPath;
	imgPath.push_back("E:\\FPCB\\linesregion\\10235\\model\\test_20190515145907_x_0_3_0_f.jpg");
	imgPath.push_back("E:\\FPCB\\linesregion\\10235\\model\\test_20190515145907_x_0_3_1_f.jpg");
	imgPath.push_back("E:\\FPCB\\linesregion\\10235\\model\\test_20190515145907_x_0_3_2_f.jpg");

	checkModel(1, imgPath);
	ImageInfo imageInfoL, imageInfoM, imageInfoR;
	imageInfoL = { imgL.data,imgL.cols, imgL.rows, imgL.step, 0 };
	imageInfoM = { imgM.data,imgM.cols, imgM.rows, imgM.step, 1 };
	imageInfoR = { imgR.data,imgR.cols, imgR.rows, imgR.step, 2 };

	//fpcb.api_loadParam("../reverse/parameter/");

	fpcb.api_learn(imageInfoL, 1);
	fpcb.api_learn(imageInfoM, 1);
	fpcb.api_learn(imageInfoR, 1);
	
	ConfigParam cparam;
	//fpcb.api_save_configparam(&cparam);
	
	OutputInfo *out = new OutputInfo();
	out->image.ptr = (unsigned char*)malloc(sizeof(unsigned char) * posData.iImgWidth * posData.iImgHeight * 2);
	out->scale = 0.5;
	out->image.height = posData.iImgHeight;
	out->image.width = posData.iImgWidth;

	fpcb.api_load_space(out);

	fpcb.api_process(imageInfoL, 1);
	fpcb.api_process(imageInfoM, 1);
	fpcb.api_process(imageInfoR, 1);
	cv::waitKey(0);


}
#include <string.h>
#define LOGGLE_COMPILER_TIME (printf("%s,%s",__FILE__,__DATE__))

int main(int argc, char **argv)
{
	//if (argc != 4)
	//{
	//	std::cout << "输入格式为:test.ext 待检测图像文件夹(不包含'/') 保存结果文件夹 正反面文件夹(0:正面 1:反面)" << std::endl;
	//	return 0;
	//}

	//process(argv[1], argv[2], argv[3]);
	run();
	//run10259();
	//run10180();
	//base_select();
	//run2();

	//run_getLines_10235();
	/*LOGGLE_COMPILER_TIME;
	printf(__FILE__); ;*/
	/*std::string a("skaj");
	for (int i=0;i<=a.length();i++)
	{
		std::cout << a[i] << std::endl;
	}*/
	return 0;
}