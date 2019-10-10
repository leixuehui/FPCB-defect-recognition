#include "detectBase.hpp"

//#define _TIME_

#define _DL
//#define SHIELD


#ifdef _TIME_LOG_|_PARAM_LOG_//由头文件进行控制
#include <Windows.h>


static std::string time_name()
{
	SYSTEMTIME time;
	GetLocalTime(&time);
	char data[11] = { 0 };
	sprintf(data, "%04d-%02d-%02d", time.wYear, time.wMonth, time.wDay);
	//std::cout << "time_defect_inspect_" + std::string(data) + ".txt";
	return  std::string(data);
}
#endif

#ifdef _TIME_LOG_
std::string log_time_name = "time_defect_inspect_" + time_name() + ".txt";
#endif

#ifdef _PARAM_LOG_
std::string log_param_name = "param_defect_inspect_" + time_name() + ".txt";
#endif


void iniLogAlg()
{
	static bool flagStatus = true;
	if (!flagStatus)
	{
		return;
	}
	flagStatus = false;

	auto console_logger = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	auto daily_logger = std::make_shared<spdlog::sinks::daily_file_sink_mt>("log_defect_inspect.txt", 2, 30);

	auto logger = std::make_shared<spdlog::logger>("log", spdlog::sinks_init_list{ console_logger, daily_logger });
	spdlog::register_logger(logger);

	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%f][thread %t][%^%l%$] %v");
	logger->flush_on(spdlog::level::info);

	std::string fileName(strrchr(__FILE__, '\\') + 1);
	std::string strTime(__TIMESTAMP__);
	logger->info("A201 Defect Inspect Logger.");
	logger->info("Current File: " + fileName + " Compile Time: " + strTime);
}

//换成类操作增加运行速度
static void classStddev(
    cv::Mat &image,
    cv::Mat &mask,
    double &foreMean,
    double &foreStddev,
    double &foreArea,
    double &backMean,
    double &backStddev,
    double &backArea)
{
    cv::Mat bin, fore, back;
    double minAreaDiff = image.cols * image.rows;
    double areaDiff;
    //int T;
    //for (int t = 1; t < 254; t++)
    //{
    //	cv::threshold(image, bin, t, 255, cv::THRESH_BINARY);
    //	fore = bin & mask;
    //	back = ~bin & mask;
    //	areaDiff = abs(sum(fore).val[0] - sum(back).val[0]) / 255;
    //	if (areaDiff < minAreaDiff)
    //	{
    //		minAreaDiff = areaDiff;
    //		T = t;
    //	}
    //}
    cv::Scalar mean;
    cv::meanStdDev(image, mean, cv::Scalar::all(0), mask);

    cv::Scalar _foreMean, _foreStddev, _backMean, _backStddev;
    cv::threshold(image, bin, mean.val[0], 255, cv::THRESH_BINARY);
    fore = bin & mask;
    back = ~bin & mask;

    cv::meanStdDev(image, _foreMean, _foreStddev, fore);
    cv::meanStdDev(image, _backMean, _backStddev, back);

    foreMean = _foreMean.val[0];
    foreStddev = _foreStddev.val[0];
    backMean = _backMean.val[0];
    backStddev = _backStddev.val[0];
    foreArea = cv::sum(fore).val[0] / 255;
    backArea = cv::sum(back).val[0] / 255;

    //diff = _foreMean.val[0] + _foreStddev.val[0] + _backStddev.val[0] - _backMean.val[0];
}

//求图像熵
static double entropy(cv::Mat &src)
{

#if _DEBUG
    assert(src.type() == CV_8UC1);
#endif

    int rows = src.rows;
    int cols = src.cols;

    double hist[256];
    for (int i = 0; i < 256; i++)
    {
        hist[i] = 0;
    }

    for (int r = 0; r < rows; r++)
    {
        uchar *ptrRow = src.ptr<uchar>(r);
        for (int c = 0; c < cols; c++)
        {
            ++hist[ptrRow[c]];
        }
    }

    int nPixels = rows*cols;
    for (int i = 0; i < 256; i++)
    {
        hist[i] /= nPixels;
    }

    double entropy = 0.0;

    for (int i = 0; i < 256; i++)
    {
        double temp = 0.0;
        if (abs(hist[i])<1e-7)
        {
            temp = 0.0;
        }
        else
        {
            temp = -log(hist[i])*hist[i] / log(2.0);

        }

        entropy += temp;

    }
    return 	entropy;
}

//圆拟合
static bool fitCircle(
    const std::vector<cv::Point> &points,
    double &cent_x,
    double &cent_y,
    double &radius)
{
    cent_x = 0.0;
    cent_y = 0.0;
    radius = 0.0;
    if (points.size() < 3)
    {
        return false;
    }

    double sum_x = 0.0, sum_y = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;
    double sum_x3 = 0.0, sum_y3 = 0.0;
    double sum_xy = 0.0, sum_x1y2 = 0.0, sum_x2y1 = 0.0;

    size_t N = points.size();
    for (int i = 0; i < N; i++)
    {
        double x = points[i].x;
        double y = points[i].y;
        double x2 = x * x;
        double y2 = y * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x2;
        sum_y2 += y2;
        sum_x3 += x2 * x;
        sum_y3 += y2 * y;
        sum_xy += x * y;
        sum_x1y2 += x * y2;
        sum_x2y1 += x2 * y;
    }

    double C, D, E, G, H;
    double a, b, c;

    C = N * sum_x2 - sum_x * sum_x;
    D = N * sum_xy - sum_x * sum_y;
    E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
    G = N * sum_y2 - sum_y * sum_y;
    H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
    a = (H * D - E * G) / (C * G - D * D);
    b = (H * C - E * D) / (D * D - G * C);
    c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

    cent_x = a / (-2);
    cent_y = b / (-2);
    radius = sqrt(a * a + b * b - 4 * c) / 2;
    return true;
}

//返回空闲内存
inline int findFreeMemory(
    bool *memoryMark,
    int memorySize)
{
    for (int i = 0; i < memorySize; i++)
        if (memoryMark[i] == false)
        {
            //std::cout<<"find:"<<i<<std::endl;
            return i;
        }
    return -1;
}

//标记内存占用
inline void lockMemory(
    bool *memoryMark,
    int n)
{
    memoryMark[n] = true;
}

//提取内存头地址
//template<typename T>
static uchar* addrMemory(
    uchar* memSpc,
    int memStep,
    int n)
{
    return memSpc + n * memStep;
}

//解锁内存占用
inline void unlockMemory(
    bool *memoryMark,
    int n)
{
    memoryMark[n] = false;
}


//提取众数
template<typename T>
static void extractMode(
    T *data,
    int start,
    int end,
    int &pos)
{
    T maxNum = 0;
    for (int i = start; i < end; i++)
    {
        if (maxNum < data[i])
        {
            maxNum = data[i];
            pos = i;
        }
    }
    return;
}

//交并比方式二维区域融合
static void fuseRect(
    std::vector<cv::Rect> &dftRoi,
    int fuseDist)
{
    //在寻找缺陷时已扩展了Rect宽度,此时只需融合有接触的roi边即可
    //融合完之后将整体搜索原先扩展的宽度即可还原位置
    if (dftRoi.size() <= 0)	return;

    int ndft = dftRoi.size();

    std::vector<cv::Rect>::iterator itc;
    cv::Point leftTop, rightBottom, leftTopItc, rightBottomItc;
    for (int n = 0; n < dftRoi.size(); n++)
    {
        leftTop = { dftRoi[n].x, dftRoi[n].y };
        rightBottom = { leftTop.x + dftRoi[n].width - 1, leftTop.y + dftRoi[n].height - 1 };
        for (itc = dftRoi.begin(); itc != dftRoi.end();)
        {
            if (n >= dftRoi.size())	break;

            if (*itc == dftRoi[n])
            {
                itc++;
                continue;
            }

            leftTopItc = { itc->x, itc->y };
            rightBottomItc = { leftTopItc.x + itc->width - 1, leftTopItc.y + itc->height - 1 };

            //判断重叠
            if (__max(leftTop.x, leftTopItc.x) - __min(rightBottom.x, rightBottomItc.x) <= 0 &&
                __max(leftTop.y, leftTopItc.y) - __min(rightBottom.y, rightBottomItc.y) <= 0)
            {
                leftTop = { __min(leftTop.x, leftTopItc.x), __min(leftTop.y, leftTopItc.y) };
                rightBottom = { __max(rightBottom.x, rightBottomItc.x), __max(rightBottom.y, rightBottomItc.y) };
                dftRoi[n].x = leftTop.x;
                dftRoi[n].y = leftTop.y;
                dftRoi[n].width = rightBottom.x - leftTop.x + 1;
                dftRoi[n].height = rightBottom.y - leftTop.y + 1;

                //{modify - 和前面的roi重叠}
                itc = dftRoi.erase(itc);
                
                if (itc - dftRoi.begin() - n < 0)
                {
                    n--;
                }

                continue;
            }
            itc++;
        }
    }

    //统一还原融合尺度
    for (int n = 0; n < dftRoi.size(); n++)
    {
        dftRoi[n].x += fuseDist;
        dftRoi[n].y += fuseDist;
        dftRoi[n].width -= fuseDist * 2;
        dftRoi[n].height -= fuseDist * 2;
    }
}


//细化
static void hilditchThin(cv::Mat& src, cv::Mat& dst)
{
    //算法有问题，得不到想要的效果
    if (src.type() != CV_8UC1)
    {
        printf("只能处理二值或灰度图像\n");
        return;
    }
    //非原地操作时候，copy src到dst
    if (dst.data != src.data)
    {
        src.copyTo(dst);
    }

    int i, j;
    int width, height;
    //之所以减2，是方便处理8邻域，防止越界
    width = src.cols - 2;
    height = src.rows - 2;
    int step = src.step;
    int  p2, p3, p4, p5, p6, p7, p8, p9;
    uchar* img;
    bool ifEnd;
    int A1;
    cv::Mat tmpimg;
    while (1)
    {
        dst.copyTo(tmpimg);
        ifEnd = false;
        img = tmpimg.data + step;
        for (i = 2; i < height; i++)
        {
            img += step;
            for (j = 2; j<width; j++)
            {
                uchar* p = img + j;
                A1 = 0;
                if (p[0] > 0)
                {
                    if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01模式
                    {
                        A1++;
                    }
                    if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01模式
                    {
                        A1++;
                    }
                    if (p[1] == 0 && p[step + 1]>0) //p4,p5 01模式
                    {
                        A1++;
                    }
                    if (p[step + 1] == 0 && p[step]>0) //p5,p6 01模式
                    {
                        A1++;
                    }
                    if (p[step] == 0 && p[step - 1]>0) //p6,p7 01模式
                    {
                        A1++;
                    }
                    if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01模式
                    {
                        A1++;
                    }
                    if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01模式
                    {
                        A1++;
                    }
                    if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01模式
                    {
                        A1++;
                    }
                    p2 = p[-step]>0 ? 1 : 0;
                    p3 = p[-step + 1]>0 ? 1 : 0;
                    p4 = p[1]>0 ? 1 : 0;
                    p5 = p[step + 1]>0 ? 1 : 0;
                    p6 = p[step]>0 ? 1 : 0;
                    p7 = p[step - 1]>0 ? 1 : 0;
                    p8 = p[-1]>0 ? 1 : 0;
                    p9 = p[-step - 1]>0 ? 1 : 0;
                    //计算AP2,AP4
                    int A2, A4;
                    A2 = 0;
                    //if(p[-step]>0)
                    {
                        if (p[-2 * step] == 0 && p[-2 * step + 1]>0) A2++;
                        if (p[-2 * step + 1] == 0 && p[-step + 1]>0) A2++;
                        if (p[-step + 1] == 0 && p[1]>0) A2++;
                        if (p[1] == 0 && p[0]>0) A2++;
                        if (p[0] == 0 && p[-1]>0) A2++;
                        if (p[-1] == 0 && p[-step - 1]>0) A2++;
                        if (p[-step - 1] == 0 && p[-2 * step - 1]>0) A2++;
                        if (p[-2 * step - 1] == 0 && p[-2 * step]>0) A2++;
                    }


                    A4 = 0;
                    //if(p[1]>0)
                    {
                        if (p[-step + 1] == 0 && p[-step + 2]>0) A4++;
                        if (p[-step + 2] == 0 && p[2]>0) A4++;
                        if (p[2] == 0 && p[step + 2]>0) A4++;
                        if (p[step + 2] == 0 && p[step + 1]>0) A4++;
                        if (p[step + 1] == 0 && p[step]>0) A4++;
                        if (p[step] == 0 && p[0]>0) A4++;
                        if (p[0] == 0 && p[-step]>0) A4++;
                        if (p[-step] == 0 && p[-step + 1]>0) A4++;
                    }
                    if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
                    {
                        if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
                        {
                            dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
                            ifEnd = true;
                            //printf("\n");

                            //PrintMat(dst);
                        }
                    }
                }
            }
        }
        //printf("\n");
        //PrintMat(dst);
        //PrintMat(dst);
        //已经没有可以细化的像素了，则退出迭代
        if (!ifEnd) break;
    }
}

SpaceParam AlgBase::gSpaceParam;
HyperParam AlgBase::gHyperParam;
HyperParam AlgBase::gHyperParamFront;
HyperParam AlgBase::gHyperParamBack;
cv::Mat AlgBase::gLut, 
AlgBase::areaLut;
PadTrain_SGM AlgBase::gMatrixPadTrain[256];
#ifndef _DEBUG
PyObject *AlgBase::pModule,
*AlgBase::pInit,
*AlgBase::pInitObj,
*AlgBase::pInfer_images,
*AlgBase::arr_images_append,
*AlgBase::arr_index_append;
#endif

AlgBase::AlgBase()
{

}

AlgBase::~AlgBase()
{

}

//初始化空间
int AlgBase::baseInit(
    int threads,
    cv::Size pcsSize,
    cv::Size imageSize)//Tip:训练的时候留最大的pcs长宽
{

    if (gSpaceParam.kernelSpace)
        free(gSpaceParam.kernelSpace);

    if (gSpaceParam.pcsSpace)
        free(gSpaceParam.pcsSpace);

    if (gSpaceParam.memoryLock)
        free(gSpaceParam.memoryLock);

    if (gSpaceParam.imageSpace)
        free(gSpaceParam.imageSpace);

    gSpaceParam.MemorySize = threads;

    //每个线程预留半径64个像素
    gSpaceParam.kernelSpaceTotalWidth = 4096;
    gSpaceParam.kernelSpace = (uchar*)malloc(4096 * sizeof(uchar) * threads);
    memset(gSpaceParam.kernelSpace, 1, 4096 * sizeof(uchar) * threads);

    gSpaceParam.pcsSpaceTotalWidth = (pcsSize.width + 100) * (pcsSize.height + 100) * 3 * sizeof(double);
    gSpaceParam.pcsSpace = (uchar*)malloc(gSpaceParam.pcsSpaceTotalWidth * threads);
    memset(gSpaceParam.pcsSpace, 0, gSpaceParam.pcsSpaceTotalWidth * threads);

    gSpaceParam.imageSpace = (uchar*)malloc(imageSize.width * imageSize.height * 3 * sizeof(uchar));
    memset(gSpaceParam.imageSpace, 0, imageSize.width * imageSize.height * 3 * sizeof(uchar));

    gSpaceParam.memoryLock = (bool*)malloc(sizeof(bool) * threads);
    memset(gSpaceParam.memoryLock, 0, sizeof(bool) * threads);

    gLut = cv::Mat::zeros(cv::Size(256, 1), CV_8UC1);
    for (int i = 0; i < 256; i++)
    {
        //线性变换
        gLut.ptr(0)[i] = cv::pow(1.f * i / 255, 0.75) * 255;
    }

    return 0;
}

int AlgBase::trainInit()
{
	for (int i=0;i<256;i++)
	{
		if (gMatrixPadTrain[i].pMatrixTrain!=nullptr)
		{
			free(gMatrixPadTrain[i].pMatrixTrain);
			gMatrixPadTrain[i].pMatrixTrain = nullptr;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		gMatrixPadTrain[i].pMatrixTrain =
			(unsigned char*)malloc(1024 * 1024 * sizeof(unsigned char));
		gMatrixPadTrain[i].rowMatrix = 0;
	}
	return 0;
}


int AlgBase::trainDestroy()
{
	for (int i = 0; i < 256; i++)
	{
		if (gMatrixPadTrain[i].pMatrixTrain != nullptr)
		{
			free(gMatrixPadTrain[i].pMatrixTrain);
			gMatrixPadTrain[i].pMatrixTrain = nullptr;
		}
	}
	return 0;
}

//传入超参
void AlgBase::loadParam(
    const HyperParam &param,
    int isTop)
{
    if (isTop == 1)
        gHyperParamFront = param;
    else
        gHyperParamBack = param;
}

//传出超参
void AlgBase::emitParam(
    HyperParam &param,
    int isTop)
{
    if (isTop == 1)
        param = gHyperParamFront;
    else
        param = gHyperParamBack;
}


int AlgBase::itemTrainer(
    const std::string &savePath,
    const cv::Mat &image,
    const ALL_REGION &gItems,
    ConfigParam &configParam,
    HyperParam &hyperParam,
    int &maxPcsWidth,
    int &maxPcsHeight/*const TrainInput & input, TrainOutput * output*/)
{

    int nPcs = gItems.size();

    if (nPcs <= 0 || image.empty())	return -1;

    cv::Mat src = image;
    //image.copyTo(src);

    //cv::Mat bgr[3];

    //cv::split(src, bgr);

    //mask收缩核
    //cv::Mat kernel(15, 15, CV_8UC1, gSpaceParam.kernelSpace);

    const PCS_REGION *pPcs;
    const ABSTRACT_REGIONS *pAbstract;

    gLut = cv::Mat::zeros(cv::Size(256, 1), CV_8UC1);
    for (int i = 0; i < 256; i++)
    {
        //线性变换
        gLut.ptr(0)[i] = cv::pow(1.f * i / 255, 0.5) * 255;
    }

	
    for (int n = 0; n < nPcs; n++)
    {
        pPcs = &gItems[n];
        for (int i = 0; i < pPcs->itemsRegions.size(); i++)
        {
            pAbstract = &pPcs->itemsRegions[i];
            switch (pAbstract->type)
            {
            case lineLay_pad:
				//padTrainer_SGM(*pAbstract, &src, gHyperParam.pad, gHyperParam.nPad, gMatrixPadTrain);
                padTrainer(*pAbstract, &src, gHyperParam.pad, lineLay_pad, gHyperParam.nPad);
                break;
            case stiffenerLay_steel:
                steelTrainer(*pAbstract, &src, gHyperParam.steel, stiffenerLay_steel, gHyperParam.nSteel);
                break;
            case printLay_EMI:
                opacityTrainer(*pAbstract, &src, gHyperParam.opacity, printLay_EMI, gHyperParam.nOpacity);
                break;
            case drillLay_position:
                break;
            case carveLay:
                carveTrainer(*pAbstract, &src, gHyperParam.carve, carveLay, gHyperParam.nCarve);
                break;
				/*case lineLay_nest:
					transprencyTrainer(*pAbstract, &src, gHyperParam.transparency, pPcs->iID, gHyperParam.nTransparency);
					break;*/
			//case lineLay_base:
			//	transprencyTrainerV2(*pAbstract, &src, gHyperParam.transparency, lineLay_base, pPcs->iID, gHyperParam.nTransparency);
			//	break;
   //         case lineLay_nest:
   //             transprencyTrainerV2(*pAbstract, &src, gHyperParam.transparency, lineLay_nest, pPcs->iID, gHyperParam.nTransparency);
   //             break;
			/*case holingThroughLay:
				holeTrainer(*pAbstract, &src, gHyperParam.hole, gHyperParam.nHole);
				break;*/
            case pcsMarkLay:
                //获取最大roi区域
                maxPcsHeight = __max(maxPcsHeight, pAbstract->items[0].mask.rows);
                maxPcsWidth = __max(maxPcsWidth, pAbstract->items[0].mask.cols);
                break;
            default:
                break;
            }
        }
    }

    hyperParam = gHyperParam;
    return 0;
}

int AlgBase::itemTrainerV2(
    const cv::Mat &image,
    const int &isTop,
    const int nImage,
    const ALL_REGION & gItems,
    TrainParam &param)
{
    int nPcs = gItems.size();

    if (nPcs <= 0 || image.empty())	return -1;

    cv::Mat src = image;

    const PCS_REGION *pPcs;

    const ABSTRACT_REGIONS *pAbstract;

    gLut = cv::Mat::zeros(cv::Size(256, 1), CV_8UC1);

    for (int i = 0; i < 256; i++)
    {
        //线性变换
        gLut.ptr(0)[i] = cv::pow(1.f * i / 255, 0.5) * 255;
    }

    for (int n = 0; n < nPcs; n++)
    {
        pPcs = &gItems[n];
        for (int i = 0; i < pPcs->itemsRegions.size(); i++)
        {
            pAbstract = &pPcs->itemsRegions[i];
            switch (pAbstract->type)
            {
            //==================Simple Train====================
            //==================    START   ====================

            case lineLay_pad:
                simpleTrain(*pAbstract, image, param, lineLay_pad);
                break;

            case stiffenerLay_steel:
                simpleTrain(*pAbstract, image, param, stiffenerLay_steel);
                break;

            case printLay_EMI:
                simpleTrain(*pAbstract, image, param, printLay_EMI);
                break;

            case carveLay:
                simpleTrain(*pAbstract, image, param, carveLay);
                break;
            
            //==================     END    ====================
            //==================Simple Train====================

            //==================Complex Train===================
            //==================    START   ====================
            case lineLay_base:
                complexTrain(*pAbstract, image, param, lineLay_base);
                break;
            case lineLay_nest:
                complexTrain(*pAbstract, image, param, lineLay_nest);
                break;
            //==================     END    ====================
            //==================Complex Train===================


            //==================Special Train===================
            //==================    START   ====================

            //==================     END    ====================
            //==================Special Train===================
            case drillLay_position:
                break;
            case pcsMarkLay:
                //获取最大roi区域
                param.maxPcsHeight = __max(param.maxPcsHeight, pAbstract->items[0].mask.rows + 100);
                param.maxPcsWidth = __max(param.maxPcsWidth, pAbstract->items[0].mask.cols + 100);
                break;
            default:
                break;
            }
        }
    }
    return 0;
}

//int AlgBase::itemInspector(
//    cv::Mat *img,
//    int isTop,
//    const PCS_REGION &gItems,
//    const ConfigParam &configParam,
//    DefectInfo &defectInfo/*const InspectorInput & input, InspectorOutput * output*/,
//    bool isModify)
//{
//
//    if (!gSpaceParam.kernelSpace ||
//        !gSpaceParam.pcsSpace ||
//        !gSpaceParam.memoryLock ||
//        !gSpaceParam.imageSpace)
//    {
//        std::cout << "第一次执行需要先加载保存文件的目录，再执行训练操作,之后执行需加载有hyperParam.xml文件的目录" << std::endl;
//        return -1;
//    }
//
//	
//    if (isTop)
//        gHyperParam = gHyperParamFront;
//    else
//        gHyperParam = gHyperParamBack;
//
//	iniLogAlg();
//	auto log = spdlog::get("log");
//
//    const PCS_REGION *pPcs = &gItems;
//    const ABSTRACT_REGIONS *pAbstract;
//
//    std::vector<cv::Rect> dftRoi;
//
//    double start;
//
//    //areaLut = cv::Mat::zeros(cv::Size(256, 1), CV_8UC1);
//    //for (int i = 0; i < 256; i++)
//    //{
//    //    //线性变换(从与之前30area断了）
//    //    areaLut.ptr(0)[i] = configParam.SteelParam.areaWeight * i + configParam.SteelParam.colorParam.infArea;// - configParam.SteelParam.areaWeight * configParam...[i]
//    //}
//
//	log->info("1/2 Start inspect. ");
//    for (int i = 0; i < pPcs->itemsRegions.size(); i++)
//    {
//        pAbstract = &pPcs->itemsRegions[i];
//        switch (pAbstract->type)
//        {
//        case lineLay_pad:
//            if (!configParam.PadParam.vaild)            break;
//            start = cv::getTickCount();
//			log->info("Pad detect inspect start. ");
//            padInspector(*pAbstract, configParam, img, dftRoi, isModify);
//			log->info("Pad detect inspect End. ");
//			log->info("Pad detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
//            //std::cout << "pad detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
//            break;
//        case stiffenerLay_steel:
//            if (!configParam.SteelParam.vaild)          break;
//            start = cv::getTickCount();
//			log->info("Steel detect inspect start. ");
//            steelInspector(*pAbstract, configParam, img, dftRoi, isModify);
//			log->info("Steel detect inspect start. ");
//			log->info("Steel detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
//            //std::cout << "steel detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
//            break;
//        case printLay_EMI:
//            if (!configParam.OpacityParam.vaild)        break;
//            start = cv::getTickCount();
//			log->info("EMI detect inspect start. ");
//            opacityInspector(*pAbstract, configParam, img, dftRoi, isModify);
//			log->info("EMI detect inspect end. ");
//            //std::cout << "emi detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
//			log->info("EMI detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
//            break;
//        case lineLay_base://现在把网格设置为基材层：08-05-2019；
//            if (!configParam.TransprencyParam.vaild)    break;
//			start = cv::getTickCount();
//			log->info("Base detect inspect start. ");
//            transprencyInspector(*pAbstract, configParam, img, dftRoi, isModify);
//			log->info("Base detect inspect start. ");
//			log->info("Base detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
//            break;
//        case holingThroughLay:
//            if (!configParam.HoleParam.vaild)           break;
//            start = cv::getTickCount();
//			log->info("Hole detect inspect start. ");
//            holeInspector(*pAbstract, configParam, img, dftRoi, isModify);
//			log->info("Hole detect inspect end. ");
//			log->info("Hole detect inspect: "+std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency())+"ms.");
//            break;
//        case carveLay:
//            if (!configParam.CarveParam.vaild)          break;
//			start = cv::getTickCount();
//			log->info("CarveLay detect inspect start. ");
//            carveInspector(*pAbstract, configParam, img, dftRoi, isModify);
//			log->info("CarveLay detect inspect end. ");
//			log->info("CarveLay detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
//			//std::cout << "CarveLay detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
//            break;
//        case charLay_fix:
//            if (!configParam.CharParam.vaild)           break;
//            start = cv::getTickCount();
//			log->info("Char detect inspect start. ");
//           charInspector(*pAbstract, configParam, img, dftRoi, isModify);
//			log->info("Char detect inspect end. ");
//			log->info("Char detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
//            //std::cout << "char detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
//            break;
//        default:
//            break;
//        }
//    }
//	log->info("1/2 End inspect. ");
//
//	log->info("2/2 Start Defects Collect. ");
//    {
//        std::lock_guard<std::mutex> _lock(muDefWrite);
//        for (int i = 0; i < dftRoi.size(); i++)
//        {
//            defectInfo.roi.push_back(dftRoi[i]);
//        }
//    }
//	log->info("2/2 End Defects Collect. ");
//    return 0;
//}

int AlgBase::itemInspector(
	cv::Mat *img,
	int isTop,
	const PCS_REGION &gItems,
	const ConfigParam &configParam,
	DefectInfo &defectInfo/*const InspectorInput & input, InspectorOutput * output*/,
	bool isModify)
{

	if (!gSpaceParam.kernelSpace    ||
		!gSpaceParam.pcsSpace       ||
		!gSpaceParam.memoryLock     ||
		!gSpaceParam.imageSpace)
	{
		std::cout << "第一次执行需要先加载保存文件的目录，再执行训练操作,之后执行需加载有hyperParam.xml文件的目录" << std::endl;
		return -1;
	}

	if (isTop)
		gHyperParam = gHyperParamFront;
	else
		gHyperParam = gHyperParamBack;

	iniLogAlg();

	auto log = spdlog::get("log");

	const PCS_REGION *pPcs = &gItems;
	const ABSTRACT_REGIONS *pAbstract;

	std::vector<DftAbstract> dftRoi;

	double start;

	std::vector<DftAbstract> defectAbstract(pPcs->itemsRegions.size());
	
	//areaLut = cv::Mat::zeros(cv::Size(256, 1), CV_8UC1);
	//for (int i = 0; i < 256; i++)
	//{
	//    //线性变换(从与之前30area断了）
	//    areaLut.ptr(0)[i] = configParam.SteelParam.areaWeight * i + configParam.SteelParam.colorParam.infArea;// - configParam.SteelParam.areaWeight * configParam...[i]
	//}

	log->info("1/2 Start inspect. ");
	for (int i = 0; i < pPcs->itemsRegions.size(); i++)
	{
		pAbstract = &pPcs->itemsRegions[i];
		switch (pAbstract->type)
		{
		case lineLay_pad:
			if (!configParam.PadParam.vaild)            break;
			start = cv::getTickCount();
			log->info("Pad detect inspect start. ");
			//padInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			//padInspector_SGM(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			defectAbstract[i].abstract = lineLay_pad;
			log->info("Pad detect inspect End. ");
			log->info("Pad detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
			//std::cout << "pad detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
			break;
		case stiffenerLay_steel:
			if (!configParam.SteelParam.vaild)          break;
			start = cv::getTickCount();
			log->info("Steel detect inspect start. ");
			//steelInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			defectAbstract[i].abstract = stiffenerLay_steel;
			log->info("Steel detect inspect start. ");
			log->info("Steel detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
			//std::cout << "steel detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
			break;
		case printLay_EMI:
			if (!configParam.OpacityParam.vaild)        break;
			start = cv::getTickCount();
			log->info("EMI detect inspect start. ");
			//opacityInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			defectAbstract[i].abstract = printLay_EMI;
			log->info("EMI detect inspect end. ");
			//std::cout << "emi detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
			log->info("EMI detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
			break;

		case lineLay_base://现在把网格设置为基材层：08-05-2019；
			if (!configParam.TransprencyParam.vaild)    break;
			start = cv::getTickCount();
			log->info("Base detect inspect start. ");
			//transprencyInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			defectAbstract[i].abstract = lineLay_base;
			log->info("Base detect inspect start. ");
			log->info("Base detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
			break;

        //case lineLay_nest://现在把网格设置为基材层：08-05-2019；
        //    if (!configParam.TransprencyParam.vaild)    break;
        //    start = cv::getTickCount();
        //    log->info("Base detect inspect start. ");
        //    nestInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
        //    defectAbstract[i].abstract = lineLay_nest;
        //    log->info("Base detect inspect start. ");
        //    log->info("Base detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
        //    break;

		case /*holingThroughLay*/drillLay_position:
			if (!configParam.HoleParam.vaild)           break;
			start = cv::getTickCount();
			log->info("Hole detect inspect start. ");
			//holeInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			defectAbstract[i].abstract = holingThroughLay;
			log->info("Hole detect inspect end. ");
			log->info("Hole detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
			break;
		case carveLay:
			if (!configParam.CarveParam.vaild)          break;
			start = cv::getTickCount();
			log->info("CarveLay detect inspect start. "); 
			//carveInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			defectAbstract[i].abstract = carveLay;
			log->info("CarveLay detect inspect end. ");
			log->info("CarveLay detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
			//std::cout << "CarveLay detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
			break;
		case charLay_fix:
			if (!configParam.CharParam.vaild)           break;
			start = cv::getTickCount();
			log->info("Char detect inspect start. ");
			//charInspector(*pAbstract, configParam, img, defectAbstract[i].rect, isModify);
			defectAbstract[i].abstract = charLay_fix;
			log->info("Char detect inspect end. ");
			log->info("Char detect inspect: " + std::to_string((cv::getTickCount() - start) * 1000 / cv::getTickFrequency()) + "ms.");
			//std::cout << "Char detect time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
			break;
		default:
			break;
		}
	}
	log->info("1/2 End inspect. ");

	log->info("2/2 Start Defects Collect. ");
	{
		std::lock_guard<std::mutex> _lock(muDefWrite);
		defectInfo.roi.swap(defectAbstract);
	}
	log->info("2/2 End Defects Collect. ");
	return 0;
}

int AlgBase::itemInspectorV2(
    const cv::Mat &img, 
    const int &isTop, 
    const int &nImage, 
    const PCS_REGION &gItems, 
    TrainParam &param,
    const ConfigParam & configParam, 
    DefectInfo & defectInfo)
{
    if (!gSpaceParam.kernelSpace ||
        !gSpaceParam.pcsSpace ||
        !gSpaceParam.memoryLock ||
        !gSpaceParam.imageSpace)
    {
        std::cout << "第一次执行需要先加载保存文件的目录，再执行训练操作,之后执行需加载有hyperParam.xml文件的目录" << std::endl;
        return -1;
    }

    iniLogAlg();

    auto log = spdlog::get("log");

    const PCS_REGION *pPcs = &gItems;

    const ABSTRACT_REGIONS *pAbstract;

    std::vector<DftAbstract> dftRoi;

    double start;

    std::vector<DftAbstract> defectAbstract(pPcs->itemsRegions.size());

    log->info("1/2 Start inspect. ");

    for (int i = 0; i < pPcs->itemsRegions.size(); i++)
    {
        pAbstract = &pPcs->itemsRegions[i];
        switch (pAbstract->type)
        {
            //==================Simple Train====================
            //==================    START   ====================
        case lineLay_pad:
            //采用深度学习方案
            if (configParam.PadParam.usingDL != 1 && configParam.PadParam.vaild)
            {
                padInspectorV2(*pAbstract, img, configParam, param, lineLay_pad, defectAbstract[i].info);
                defectAbstract[i].abstract = lineLay_pad;
            }
            break;

        case stiffenerLay_steel:
            if (configParam.SteelParam.usingDL != 1 && configParam.SteelParam.vaild)
            {
                steelInspectorV2(*pAbstract, img, configParam, param, stiffenerLay_steel, defectAbstract[i].info);
                defectAbstract[i].abstract = stiffenerLay_steel;
            }
            break;

        case printLay_EMI:
            if (configParam.OpacityParam.usingDL != 1 && configParam.OpacityParam.vaild)
            {
                emiInspectorV2(*pAbstract, img, configParam, param, printLay_EMI, defectAbstract[i].info);
                defectAbstract[i].abstract = printLay_EMI;
            }
            break;

        case carveLay:
            if (configParam.CarveParam.usingDL != 1 && configParam.CarveParam.vaild)
            {
                carveInspectorV2(*pAbstract, img, configParam, param, carveLay, defectAbstract[i].info);
                defectAbstract[i].abstract = carveLay;
            }
            break;

            //==================     END    ====================
            //==================Simple Inspector================

            //==================Complex Inspector===============
            //==================    START   ====================
        case lineLay_base:
            if (configParam.TransprencyParam.usingDL != 1 && configParam.TransprencyParam.vaild)
            {
                complexInspector(*pAbstract, img, configParam, param, lineLay_base, defectAbstract[i].info);
                defectAbstract[i].abstract = lineLay_base;
            }
            break;
        case lineLay_nest:
            if (configParam.TransprencyParam.usingDL != 1 && configParam.TransprencyParam.vaild)
            {
                complexInspector(*pAbstract, img, configParam, param, lineLay_nest, defectAbstract[i].info);
                defectAbstract[i].abstract = lineLay_nest;
            }
            break;
            //==================     END    ====================
            //==================Complex Inspector===============
        
            //==================Special Inspector===============
            //==================    START   ====================
        case drillLay_position:
            break;
        case charLay_fix:
            break;
            //==================     END    ====================
            //==================Special Inspector===============
        }
    }

    log->info("1/2 End inspect. ");

    log->info("2/2 Start Defects Collect. ");

    {
        std::lock_guard<std::mutex> _lock(muDefWrite);
        defectInfo.roi.swap(defectAbstract);
    }

    log->info("2/2 End Defects Collect. ");
    return 0;
}

#ifndef _DEBUG && _DL

//提取缺陷数据
template<typename T>
inline void getElement(PyObject *ListItem, int i, T &dst)
{
    PyObject *Item = PyList_GetItem(ListItem, i);
    if (typeid(T) == typeid(int))
        dst = PyLong_AsLong(Item);
    else if (typeid(T) == typeid(double))
        dst = PyFloat_AsDouble(Item);
    //Py_DECREF(Item); //释放空间(第二次运行会崩）
}

static wchar_t* stringToWchar(const std::string &c)
{
	int len = MultiByteToWideChar(CP_ACP, 0, c.c_str(), strlen(c.c_str()), NULL, 0);
	wchar_t* m_wchar = new wchar_t[len + 1];
	MultiByteToWideChar(CP_ACP, 0, c.c_str(), strlen(c.c_str()), m_wchar, len);
	m_wchar[len] = '\0';
	return m_wchar;
}

int AlgBase::cudaInit()
{
	static bool flag = false;
	if (flag)
	{
		return 0;
	}

	std::ifstream file("path.txt");

	if (!file)
	{
		printf("DL PATH ISN'T EXIT OR NO READING ACCEESS!!!\r\n");
		return -1;
	}
	std::string pathPH;
	std::string path1;
	std::string path2;
	getline(file, pathPH);
	getline(file, path1);
	getline(file, path2);

	std::cout << pathPH << std::endl;
	std::cout << path1 << std::endl;
	std::cout << path2 << std::endl;
	//pathPH = "D:/Program Files/Anaconda3/envs/pytorch";
	wchar_t *pathPythonHome = stringToWchar(pathPH);
	Py_SetPythonHome(pathPythonHome);
	//Py_SetPythonHome(L"D:/Program Files/Anaconda3/envs/pytorch");
    Py_Initialize();    //初始化python解释器；
	if (!Py_IsInitialized())//检查python解释器是否成功初始化；
	{
		return DETECT_CUDA_INIT_ERROR; 
	}
		

    try
    {
        PyRun_SimpleString("print('Python Start')");//执行一段python代码；
        PyRun_SimpleString("import sys");
		PyRun_SimpleString(path1.c_str());
		PyRun_SimpleString(path2.c_str());
        //PyRun_SimpleString("sys.path.append('F:/VSProgram/FPCB07-16/DL/fpcb_interface_bak/fpcb_interface/Detectron-master-mutiprocess')"); //这句不加会导致import 错误， 导致init错误， 错误往上抛出， 就直接退出了
        //PyRun_SimpleString("sys.path.append('F:/VSProgram/FPCB07-16/DL/fpcb_interface_bak/fpcb_interface/Detectron-master-mutiprocess/tools')"); //vs 的工作目录在
        //PyRun_SimpleString("print(sys.path)");
    }
    catch (...)
    {

        std::printf("Error to import python!");
		return -1;
    }

    pModule = NULL;
    pInit = NULL;
    pInitObj = NULL;
    pInfer_images = NULL,
    arr_images_append = NULL,
    arr_index_append = NULL;

	/*
	PyObject* PyImport_ImportModule(char *name)
	导入一个Python模块，参数name可以是*.py文件的文件名。类似Python内建函数import。
	*/
	
    pModule = PyImport_ImportModule("pipe2");

	if (pModule == NULL)                
		return DETECT_CUDA_MODULE_ERROR;

	/*
	PyObject* PyObject_GetAttrString(PyObject *o, char*attr_name)
    返回模块对象o中的attr_name 属性或函数，相当于Python中表达式语句，o.attr_name。
	*/
    pInit = PyObject_GetAttrString(pModule, "init");
    pInitObj = PyObject_CallObject(pInit, NULL);//调用python函数；

    pInfer_images = PyObject_GetAttrString(pModule, "infer_images");
    if (pInfer_images == NULL)          return DETECT_CUDA_INST_ERROR;

    arr_images_append = PyObject_GetAttrString(pModule, "append_images");
    if (arr_images_append == NULL)      return DETECT_CUDA_INST_ERROR;

    arr_index_append = PyObject_GetAttrString(pModule, "append_index");
    if (arr_index_append == NULL)       return DETECT_CUDA_INST_ERROR;

    flag = true;
    return 0;
}
#endif

#ifndef _DEBUG && _DL
struct DefInfoCuda
{
    int ltx; int lty;
    int rbx; int rby;
    double cfd;
    int pcsId;
};
struct PcsInfo
{
    cv::Mat image;
    int n;
};

inline 
bool iou(const cv::Rect &templ, 
	const cv::Rect &obj, 
	cv::Rect &cross)
{
    int ltx = __max(templ.x, obj.x),
        lty = __max(templ.y, obj.y),
        rbx = __min(templ.x + templ.width - 1, obj.x + obj.width - 1),
        rby = __min(templ.y + templ.height - 1, obj.y + obj.height - 1);
    if (ltx - rbx < 0 && lty - rby < 0)
    {
        //获取检测ROI与mask交叠的位置
        cross.x = ltx - obj.x;
        cross.y = lty - obj.y;
        cross.width = rbx - ltx + 1;
        cross.height = rby - lty + 1;
        return true;
    }
    return false;
}
//int AlgBase::itemInspectorCuda(
//	cv::Mat *img,
//	const ALL_REGION &gItems,
//	ConfigParam configParam,
//	std::vector<DefectCudaInfo> &defectInfo)
//{
//	if (!configParam.usingDL())  return 0;
//
//	std::cout << __LINE__ << std::endl;
//	std::vector<PcsInfo> array_images;
//	std::vector<DefInfoCuda> defects;
//	const ABSTRACT_REGIONS	*pAbstract = NULL;
//	const ITEM_REGION		*pItem = NULL;
//	//const PCS_REGION        *pPcs = NULL;
//	std::cout << __LINE__ << std::endl;
//	import_array();
//	PyByteArrayObject *pyIMgArr;
//	PyObject *PyImgObj;
//	PyObject *res;
//
//	cv::Mat mask;
//	cv::Mat subImg;
//	if (img->empty())   return -1;
//
//	//items的起始结束区域
//	int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
//	int dx = 0, dy = 0;
//	cv::Rect roi, maskRoi, defRoi;
//
//	double areaRate;
//
//	int nPcs = gItems.size();
//
//	std::cout << __LINE__ << std::endl;
//	//提取pcs
//
//	static int a = 0;
//	int acc = 0;
//	for (int n = 0; n < nPcs; n++)
//	{
//		for (int m = 0; m < gItems[n].itemsRegions.size(); m++)
//		{
//			pAbstract = &(gItems[n].itemsRegions[m]);
//
//			if (pAbstract->type != Layer::pcsContourLay)
//				continue;
//
//			pItem = &(pAbstract->items[0]);
//
//			//cv::cvtColor(pItem->mask, mask, cv::COLOR_GRAY2BGR);
//			mask = pItem->mask;
//
//			sx = __max(0, pItem->iOffsetX);
//			sy = __max(0, pItem->iOffsetY);
//			ex = __min(pItem->iOffsetX + mask.cols, img->cols);
//			ey = __min(pItem->iOffsetY + mask.rows, img->rows);
//			width = ex - sx;
//			height = ey - sy;
//			roi = { sx, sy, width, height };
//			if (ex <= 0 || width <= 0)	continue;
//
//			maskRoi = { sx - pItem->iOffsetX,
//				0,
//				width,
//				height };
//
//			subImg = (*img)(roi).clone();
//			array_images.push_back({ subImg,gItems[n].iID });
//			
//			cv::imwrite(std::to_string(a) + "_" + std::to_string(acc) + ".jpg", subImg);
//			acc++;
//		}
//	}
//	a++;
//	std::cout << __LINE__ << std::endl;
//	//数组转入python端
//
//	int block = array_images.size() / 20;
//	
//	for (int i = 0; i < array_images.size(); i++)
//	{
//		/*if (i<20)*/
//		{
//			size_t img_size = array_images[i].image.total() * array_images[i].image.elemSize();
//			npy_intp IMGSHAPE[3] = { array_images[i].image.rows, array_images[i].image.cols, array_images[i].image.channels() };
//			pyIMgArr = reinterpret_cast<PyByteArrayObject *>(PyArray_SimpleNewFromData(3, IMGSHAPE, NPY_UBYTE, reinterpret_cast<void *>(array_images[i].image.data)));
//			PyImgObj = reinterpret_cast<PyObject *>(pyIMgArr);
//
//			PyObject* arg_image = PyTuple_New(1);
//			PyTuple_SetItem(arg_image, 0, PyImgObj);
//
//			PyObject* arg_index = PyTuple_New(1);
//			PyTuple_SetItem(arg_index, 0, PyLong_FromLong(array_images[i].n));
//			PyObject_CallObject(arr_index_append, arg_index);
//			PyObject_CallObject(arr_images_append, arg_image);
//			Py_DECREF(arg_image);
//			Py_DECREF(arg_index);
//		}
//		
//	}
//	std::cout << __LINE__ << std::endl;
//	PyObject *pyArgs = PyTuple_New(1);
//	PyTuple_SetItem(pyArgs, 0, pInitObj);
//	std::cout << __LINE__ << std::endl;
//	res = PyObject_CallObject(pInfer_images, pyArgs);
//	std::cout << __LINE__ << std::endl;
//	if (res != NULL)
//	{
//		int SizeOfList = PyList_Size(res);
//		defects.clear();
//		for (int Index_i = 0; Index_i < SizeOfList; Index_i++)
//		{
//			PyObject *ListItem = PyList_GetItem(res, Index_i);
//
//			//获取List对象中的每一个元素
//			int NumOfItems = PyList_Size(ListItem);
//
//			//List对象子元素的大小，这里NumOfItems = 3 
//			DefInfoCuda def;
//			getElement<int>(ListItem, 0, def.ltx);
//			getElement<int>(ListItem, 1, def.lty);
//			getElement<int>(ListItem, 2, def.rbx);
//			getElement<int>(ListItem, 3, def.rby);
//			getElement<double>(ListItem, 4, def.cfd);
//			getElement<int>(ListItem, 5, def.pcsId);
//			defects.push_back(def);
//			Py_DECREF(ListItem);//释放空间 
//		}
//	}
//	std::cout << __LINE__ << std::endl;
//	std::cout << "Before Selectcted::" << defects.size() << std::endl;
//	int numDefectSelect = 0;
//	for (int i = 0; i < defects.size(); i++)
//	{
//
//		for (int n = 0; n < nPcs; n++)//n；PCS NO.;
//		{
//			if (gItems[n].iID != defects[i].pcsId /*|| defects[i].cfd < 0.5*/)   //判断缺陷属于哪一个PCS;
//				continue;
//
//
//			/*flagPcs = true;*/
//			for (int m = 0; m < gItems[n].itemsRegions.size(); m++)
//			{
//				pAbstract = &gItems[n].itemsRegions[m];
//				if (pAbstract->type == Layer::pcsContourLay)
//				{
//					pItem = &pAbstract->items[0];
//					dx = __max(0, pItem->iOffsetX);
//					dy = __max(0, pItem->iOffsetY);
//				}
//			}
//			for (int m = 0; m < gItems[n].itemsRegions.size(); m++)//通过是否使用深度学习来判断其可能的几个属性；
//			{
//				pAbstract = &gItems[n].itemsRegions[m];
//
//				switch (pAbstract->type)
//				{
//				case lineLay_pad:
//					if (!configParam.PadParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case stiffenerLay_steel:
//					if (!configParam.SteelParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case printLay_EMI:
//					if (!configParam.OpacityParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case lineLay_conduct:
//					if (!configParam.LineParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case carveLay:
//					if (!configParam.CarveParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case lineLay_finger:
//					if (!configParam.FingerParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case lineLay_baojiao:
//					if (!configParam.TransprencyParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case printLay_lvyou:
//					if (!configParam.TransprencyParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case lineLay_base:
//					if (!configParam.TransprencyParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				case lineLay_nest:
//					if (!configParam.TransprencyParam.usingDL)
//					{
//						continue;
//					}
//					break;
//
//				default:
//					continue;
//					break;
//				}
//
//				for (int k = 0; k < pAbstract->items.size(); k++)
//				{
//					pItem = &(pAbstract->items[k]);
//
//					mask = pItem->mask;
//
//					//提取当前mask对于整个pcs的相对位置
//					sx = __max(0, pItem->iOffsetX - dx);
//					sy = __max(0, pItem->iOffsetY - dy);
//					ex = __min(pItem->iOffsetX - dx + mask.cols, img->cols);
//					ey = __min(pItem->iOffsetY - dy + mask.rows, img->rows);
//					width = ex - sx;
//					height = ey - sy;
//
//					maskRoi = { sx, sy, width, height };
//					//mask未出现在图片上剔除
//					if (ex <= 0 || width <= 0)
//						continue;
//
//					//mask与缺陷最大外接矩形没有交集剔除
//					defRoi = { 0,0,0,0 };
//					roi = { defects[i].ltx, defects[i].lty, defects[i].rbx - defects[i].ltx + 1, defects[i].rby - defects[i].lty + 1 };
//					//std::cout<<roi.x<<","<<roi.y<<","<<roi.width<<","<<roi.height<<std::endl;
//					//std::cout << maskRoi.x << "," << maskRoi.y << "," << maskRoi.width << "," << maskRoi.height << std::endl;
//					//std::cout << defRoi.x << "," << defRoi.y << "," << defRoi.width << "," << defRoi.height << std::endl;
//					if (!iou(roi, maskRoi, defRoi))
//						continue;
//
//					//判断缺陷落入item的位置
//					areaRate = cv::sum(mask(defRoi)).val[0] / (255 * roi.width * roi.height);
//
//					if (areaRate > 0.3)
//					{
//						roi.x += dx;
//						roi.y += dy;
//						//vecDefects[gItems[n].iID].abstract = Layer::lineLay_pad;
//						defectInfo[defects[n].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type ,pItem->iID,roi });
//						/*flagAbstract = true;*/
//						numDefectSelect++;
//						break; //当找到缺陷所属在定位的属性层后，跳出当前属性层的其余项的循环；
//					}
//
//				}
//
//			}
//		}
//	}
//
//	std::cout << "After Selectcted::" << numDefectSelect << std::endl;
//	std::cout << __LINE__ << std::endl;
//	return 0;
//}

//old version;
//itemInspectorCuda;

//#define IMGSAVECUDA
int AlgBase::itemInspectorCuda(cv::Mat *img, const ALL_REGION &gItems, ConfigParam configParam, std::vector<DefectCudaInfo> &defectInfo)
{

	iniLogAlg();
	auto log = spdlog::get("log");
	std::string strFunction(__FUNCTION__);


    if (!configParam.usingDL())  return 0;


#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point timeStart = std::chrono::steady_clock::now();

#endif

    std::vector<PcsInfo> array_images;
    std::vector<DefInfoCuda> defects;
    const ABSTRACT_REGIONS	*pAbstract = NULL;
    const ITEM_REGION		*pItem = NULL;
    //const PCS_REGION        *pPcs = NULL;
    import_array();
    PyByteArrayObject *pyIMgArr;
    PyObject *PyImgObj;
    PyObject *res;

    cv::Mat mask;
    cv::Mat subImg;
    if (img->empty())   return -1;
    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    int dx = 0, dy = 0;
    cv::Rect roi, maskRoi, defRoi;

    double areaRate;
    int nPcs = gItems.size();
    //提取pcs
	

	double areaMax;
	double widthMax;
	double heightMax;

	cv::RotatedRect dftRtRect ;
	cv::Rect dftRect;
	double dftArea ;
	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(61, 61));
	

#ifdef IMGSAVECUDA
	static int num = 0;
	int acc = 0;
	std::vector<int> vecID;
	std::vector<cv::Point> vecPtTl;

#endif

#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
	std::ofstream file(log_time_name, std::ios::out | std::ios::app);
	std::chrono::milliseconds timeCostSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - timeStart);
	std::chrono::milliseconds timeCostFull = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - timeStart);
	file << "    "<<std::left << std::setw(32) <<strFunction + "1 输入检查"
		<< std::right << std::setw(16) << timeCostFull.count()
		<< std::right << std::setw(16) << timeCostSingle.count() 
		<< std::endl;
#endif

    for (int n = 0; n < nPcs; n++)
    {
        for (int m = 0; m < gItems[n].itemsRegions.size(); m++)
        {
            pAbstract = &(gItems[n].itemsRegions[m]);
            if (pAbstract->type != Layer::pcsContourLay)    continue;

            pItem = &(pAbstract->items[0]);

            //cv::cvtColor(pItem->mask, mask, cv::COLOR_GRAY2BGR);
            mask = pItem->mask;

            sx = __max(0, pItem->iOffsetX);
            sy = __max(0, pItem->iOffsetY);
            ex = __min(pItem->iOffsetX + mask.cols, img->cols);
            ey = __min(pItem->iOffsetY + mask.rows, img->rows);
            width = ex - sx;
            height = ey - sy;
            roi = { sx, sy, width, height };
            if (ex <= 0 || width <= 0)	continue;

            maskRoi = { sx - pItem->iOffsetX,
                0,
                width,
                height };


			cv::morphologyEx((mask)(maskRoi),
				(mask)(maskRoi),
				cv::MorphTypes::MORPH_ERODE,
				kernel,
				cv::Point(-1, -1),
				1, cv::BorderTypes::BORDER_CONSTANT, 0);

           
			(*img)(roi).copyTo(subImg, mask(maskRoi));


            array_images.push_back({ subImg,gItems[n].iID});

#ifdef IMGSAVECUDA
			vecID.push_back(gItems[n].iID);
			vecPtTl.push_back(cv::Point(sx, sy));
			cv::imwrite(".\\PCS\\"+std::to_string(num)+"_"+std::to_string(acc) + ".jpg", subImg);
			//cv::imwrite(".\\PCS\\_" + std::to_string(num) + "_" + std::to_string(acc) + ".jpg", (*img)(roi));
			//cv::imwrite(".\\PCS\\__" + std::to_string(num) + "_" + std::to_string(acc) + ".jpg", mask(maskRoi));
			acc++;
			log->info("--------CUDA 切割保存完毕--------");
#endif
        }
    }


#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	timeCostSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
	timeCostFull = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - timeStart);
	file <<"    "<< std::left << std::setw(32) <<strFunction + "2 图像切割"
		<< std::right << std::setw(16) << timeCostFull.count()
		<< std::right << std::setw(16) << timeCostSingle.count() 
		<< std::endl;
#endif
	
	//nImg++;
    //数组转入python端
    for (int i = 0; i < array_images.size(); i++)
    {
        size_t img_size = array_images[i].image.total() * array_images[i].image.elemSize();
        npy_intp IMGSHAPE[3] = { array_images[i].image.rows, array_images[i].image.cols, array_images[i].image.channels() };
        pyIMgArr = reinterpret_cast<PyByteArrayObject *>(PyArray_SimpleNewFromData(3, IMGSHAPE, NPY_UBYTE, reinterpret_cast<void *>(array_images[i].image.data)));
        PyImgObj = reinterpret_cast<PyObject *>(pyIMgArr);

        PyObject* arg_image = PyTuple_New(1);
        PyTuple_SetItem(arg_image, 0, PyImgObj);

        PyObject* arg_index = PyTuple_New(1);
        PyTuple_SetItem(arg_index, 0, PyLong_FromLong(array_images[i].n));
        PyObject_CallObject(arr_index_append, arg_index);
        PyObject_CallObject(arr_images_append, arg_image);
        Py_DECREF(arg_image);
        Py_DECREF(arg_index);
    }
    PyObject *pyArgs = PyTuple_New(1);
    PyTuple_SetItem(pyArgs, 0, pInitObj);
	
#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time3 = std::chrono::steady_clock::now();
	timeCostSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2);
	timeCostFull = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - timeStart);
	
	file <<"    "<< std::left << std::setw(32) <<strFunction + "3 数据转换"
		<< std::right << std::setw(16) << timeCostFull.count()
		<< std::right << std::setw(16) << timeCostSingle.count() 
		<< std::endl;
#endif

    res = PyObject_CallObject(pInfer_images, pyArgs);
    if (res != NULL)
    {
        int SizeOfList = PyList_Size(res);
        defects.clear();
        for (int Index_i = 0; Index_i < SizeOfList; Index_i++)
        {
            PyObject *ListItem = PyList_GetItem(res, Index_i);

            //获取List对象中的每一个元素
            int NumOfItems = PyList_Size(ListItem);

            //List对象子元素的大小，这里NumOfItems = 3 
            DefInfoCuda def;
            getElement<int>(ListItem, 0, def.ltx);
            getElement<int>(ListItem, 1, def.lty);
            getElement<int>(ListItem, 2, def.rbx);
            getElement<int>(ListItem, 3, def.rby);
            getElement<double>(ListItem, 4, def.cfd);
            getElement<int>(ListItem, 5, def.pcsId);
            defects.push_back(def);
            Py_DECREF(ListItem);//释放空间 
        }
    }
#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time4 = std::chrono::steady_clock::now();
	timeCostSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3);
	timeCostFull = std::chrono::duration_cast<std::chrono::milliseconds>(time4 - timeStart);
	
	file << "    "<<std::left << std::setw(32) <<strFunction + "4 缺陷获取"
		<< std::right << std::setw(16) << timeCostFull.count()
		<< std::right << std::setw(16) << timeCostSingle.count() 
		<< std::endl;
#endif
	log->info("Before Selected: "+ defects.size());

#ifdef IMGSAVECUDA

	
	cv::Mat imgShow = (*img).clone();
	cv::cvtColor(imgShow, imgShow, cv::ColorConversionCodes::COLOR_BGR2RGB);
	
	for (int i = 0; i < defects.size(); i++)
	{
		for (int n = 0; n < vecPtTl.size(); n++)
		{
			//std::cout << n << ":" << gItems[n].itemsRegions.size() << ",";
			if (vecID[n] != defects[i].pcsId)   continue;

			cv::rectangle(imgShow, cv::Rect(cv::Point(defects[i].ltx + vecPtTl[n].x, defects[i].lty + vecPtTl[n].y), cv::Point(defects[i].rbx + vecPtTl[n].x, defects[i].rby + vecPtTl[n].y)), cv::Scalar(0, 255, 0), 1);
			cv::putText(imgShow, std::to_string(i)+": "+ std::to_string(defects[i].cfd), cv::Point(defects[i].ltx, defects[i].lty - 8)  + vecPtTl[n], 1, 1, cv::Scalar(0, 255, 0),1);

		}
	}

	cv::putText(imgShow, "Total defects before selecting: " + std::to_string(defects.size()), cv::Point( 100,100), 1, 5, cv::Scalar(0, 255, 0), 3);
	
#endif
	
    //缺陷筛选
	int numDefectSelect = 0;
    for (int i = 0; i < defects.size(); i++)
    {
		/*if (defects[i].cfd < 0.99) continue;*/
        for (int n = 0; n < nPcs; n++)
        {
            if (gItems[n].iID != defects[i].pcsId)   continue;
			
            //当前pcs的偏移量；
            for (int m = 0; m < gItems[n].itemsRegions.size(); m++)
            {
                pAbstract = &gItems[n].itemsRegions[m];
                if(pAbstract->type == Layer::pcsContourLay)
                {
                    pItem = &pAbstract->items[0];
                    dx = __max(0,pItem->iOffsetX);
                    dy = __max(0,pItem->iOffsetY);
                }
            }

            for (int m = 0; m < gItems[n].itemsRegions.size(); m++)
            {
                pAbstract = &gItems[n].itemsRegions[m];
                //std::cout<<pAbstract->type<<std::endl;

                if (pAbstract->type != Layer::lineLay_pad           && 
                    pAbstract->type != Layer::stiffenerLay_steel    &&
                    pAbstract->type != Layer::printLay_EMI          &&
                    pAbstract->type != Layer::lineLay_conduct       &&
                    pAbstract->type != Layer::carveLay              &&
                    pAbstract->type != Layer::lineLay_finger        &&
                    pAbstract->type != Layer::lineLay_baojiao       && 
                    pAbstract->type != Layer::printLay_lvyou        &&
                    pAbstract->type != Layer::lineLay_base          &&
                    pAbstract->type != Layer::lineLay_nest)
                continue;

                for (int k = 0; k < pAbstract->items.size(); k++)
                {
                    pItem = &(pAbstract->items[k]);

                    mask = pItem->mask;
                    
                    //提取当前mask对于整个pcs的相对位置
                    sx = __max(0, pItem->iOffsetX - dx);
                    sy = __max(0, pItem->iOffsetY - dy);
                    ex = __min(pItem->iOffsetX - dx + mask.cols, img->cols);
                    ey = __min(pItem->iOffsetY - dy + mask.rows, img->rows);
                    width = ex - sx;
                    height = ey - sy;
                    
                    maskRoi = { sx, sy, width, height };
                    //mask未出现在图片上剔除
                    if (ex <= 0 || width <= 0)	continue;

                    //mask与缺陷最大外接矩形没有交集剔除
                    defRoi = {0,0,0,0};
                    roi = {defects[i].ltx, defects[i].lty, defects[i].rbx - defects[i].ltx + 1, defects[i].rby - defects[i].lty + 1};
                    //std::cout<<roi.x<<","<<roi.y<<","<<roi.width<<","<<roi.height<<std::endl;
                    //std::cout << maskRoi.x << "," << maskRoi.y << "," << maskRoi.width << "," << maskRoi.height << std::endl;
                    //std::cout << defRoi.x << "," << defRoi.y << "," << defRoi.width << "," << defRoi.height << std::endl;
                    if (!iou(roi, maskRoi, defRoi))  continue;
                    
                    //判断缺陷落入item的位置
                    areaRate = cv::sum(mask(defRoi)).val[0] / (255 * roi.width * roi.height);
                    
                    if (areaRate > 0.5)
                    {
						if ((pAbstract->type == Layer::lineLay_pad && configParam.PadParam.usingDL) ||
							(pAbstract->type == Layer::stiffenerLay_steel && configParam.SteelParam.usingDL) ||
							(pAbstract->type == Layer::printLay_EMI && configParam.OpacityParam.usingDL) ||
							(pAbstract->type == Layer::lineLay_conduct && configParam.LineParam.usingDL) ||
							(pAbstract->type == Layer::carveLay && configParam.CarveParam.usingDL) ||
							(pAbstract->type == Layer::lineLay_finger && configParam.FingerParam.usingDL) ||
							(pAbstract->type == Layer::lineLay_baojiao && configParam.TransprencyParam.usingDL) ||
							(pAbstract->type == Layer::printLay_lvyou && configParam.TransprencyParam.usingDL) ||
							(pAbstract->type == Layer::lineLay_base && configParam.TransprencyParam.usingDL) ||
							(pAbstract->type == Layer::lineLay_nest && configParam.TransprencyParam.usingDL))
						{
							//roi.x += dx;
							//roi.y += dy;
							//defectInfo[defects[i].pcsId].dft.push_back({gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi});
							
#ifdef IMGSAVECUDA
							cv::rectangle(imgShow, cv::Rect(cv::Point(roi.x + dx, roi.y + dy), cv::Size(roi.width, roi.height)), cv::Scalar(0, 255, 255), 1);
							cv::putText(imgShow, "Index: " + std::to_string(numDefectSelect) + "Abstract:" + std::to_string(pAbstract->type), cv::Point(roi.x + dx ,dy + roi.y +roi.height + 10), 1, 1, cv::Scalar(0, 255, 255), 1);

#endif // IMGSAVECUDA

							switch (pAbstract->type)
							{
							case Layer::lineLay_pad:
							{
								if (defects[i].cfd < configParam.PadParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);
#endif

							}
							break;

							case Layer::stiffenerLay_steel:
							{
								if (defects[i].cfd < configParam.SteelParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif
							
							}
							break;

							case Layer::printLay_EMI:
							{
								if (defects[i].cfd < configParam.OpacityParam.cfg)
								{
									continue;
								}

								subImg = (*img)({ dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }).clone();

								cv::cvtColor(subImg, subImg, cv::ColorConversionCodes::COLOR_BGR2GRAY);
								cv::threshold(subImg, subImg, 0, 255, cv::ThresholdTypes::THRESH_OTSU);

								kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, 
									cv::Size(configParam.OpacityParam.shrinkSize * 2 + 1, configParam.OpacityParam.shrinkSize * 2 + 1));

								cv::morphologyEx((mask)(defRoi),
									(mask)(defRoi),
									cv::MorphTypes::MORPH_ERODE,
									kernel,
									cv::Point(-1, -1),
									1, cv::BorderTypes::BORDER_CONSTANT, 0);

								subImg &= (mask)(defRoi);

								std::vector<std::vector<cv::Point>> vecContours;
								cv::findContours(subImg, vecContours, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);

								areaMax = -1.0;
								widthMax = -1.0;
								heightMax = -1.0;

								for (int c = 0; c < vecContours.size(); c++)
								{
									dftRtRect = cv::minAreaRect(vecContours[c]);
									dftRect = cv::boundingRect(vecContours[c]);
									//double dftArea = cv::sum(imgEMI).val[0] / 255;

									dftArea = cv::contourArea(vecContours[c]);
#ifdef IMGSAVECUDA
									cv::putText(imgShow, "Index: " + std::to_string(c) + "Area: " + std::to_string(dftArea) + "Width: " + std::to_string(dftRtRect.size.width) + "Height: " + std::to_string(dftRtRect.size.height),
										cv::Point(dx + sx + defRoi.x, dy + sy + defRoi.y - 20-20 * c), 1, 1, cv::Scalar(255, 255, 255), 1);
									if (dftArea > configParam.OpacityParam.colorParam.infArea ||
										dftRtRect.size.width > configParam.OpacityParam.colorParam.infWidth ||
										dftRtRect.size.height > configParam.OpacityParam.colorParam.infHeight)

									{

										cv::putText(imgShow, "Index: " + std::to_string(c) + "Area: " + std::to_string(dftArea) + "Width: " + std::to_string(dftRtRect.size.width) + "Height: " + std::to_string(dftRtRect.size.height),
											cv::Point(dx + sx + defRoi.x, dy + sy + defRoi.y - 20- 20 * c), 1, 1, cv::Scalar(0, 0, 255), 2);
									}

#endif // IMGSAVECUDA
									if (dftRtRect.size.width>widthMax)
									{
										widthMax = dftRtRect.size.width;
									}
									if (dftRtRect.size.height > heightMax)
									{
										heightMax = dftRtRect.size.height;
									}
									if (dftArea > areaMax)
									{
										areaMax = dftArea;
									}
								}
								if (areaMax > configParam.OpacityParam.colorParam.infArea ||
									widthMax > configParam.OpacityParam.colorParam.infWidth ||
									heightMax > configParam.OpacityParam.colorParam.infHeight)

								{

									roi.x += dx;
									roi.y += dy;
									defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
									cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif
								}

							
							
							}
								break;

							case Layer::lineLay_conduct:
							{
								if (defects[i].cfd < configParam.LineParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif
							
							}
							break;

							case Layer::carveLay:
							{
								if (defects[i].cfd < configParam.CarveParam.cfg)
								{
									continue;
								}

								subImg = (*img)({ dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }).clone();

								cv::cvtColor(subImg, subImg, cv::ColorConversionCodes::COLOR_BGR2GRAY);
								cv::threshold(subImg, subImg, 0, 255, cv::ThresholdTypes::THRESH_OTSU);

								kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT,
									cv::Size(configParam.CarveParam.shrinkSize * 2 + 1, configParam.CarveParam.shrinkSize * 2 + 1));

								cv::morphologyEx((mask)(defRoi),
									(mask)(defRoi),
									cv::MorphTypes::MORPH_ERODE,
									kernel,
									cv::Point(-1, -1),
									1, cv::BorderTypes::BORDER_CONSTANT, 0);

								subImg &= (mask)(defRoi);

								std::vector<std::vector<cv::Point>> vecContours;
								cv::findContours(subImg, vecContours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);

								areaMax = -1.0;
								widthMax = -1.0;
								heightMax = -1.0;

								for (int c = 0; c < vecContours.size(); c++)
								{
									dftRtRect = cv::minAreaRect(vecContours[c]);
									dftRect = cv::boundingRect(vecContours[c]);
									//double dftArea = cv::sum(imgEMI).val[0] / 255;

									dftArea = cv::contourArea(vecContours[c]);
#ifdef IMGSAVECUDA
									cv::putText(imgShow, "Index: " + std::to_string(c) + "Area: " + std::to_string(dftArea) + "Width: " + std::to_string(dftRtRect.size.width) + "Height: " + std::to_string(dftRtRect.size.height),
										cv::Point(dx + sx + defRoi.x, dy + sy + defRoi.y - 20 - 20 * c), 1, 1, cv::Scalar(255, 255, 255), 1);
									if (dftArea > configParam.CarveParam.colorParam.infArea ||
										dftRtRect.size.width > configParam.CarveParam.colorParam.infWidth ||
										dftRtRect.size.height > configParam.CarveParam.colorParam.infHeight)

									{

										cv::putText(imgShow, "Index: " + std::to_string(c) + "Area: " + std::to_string(dftArea) + "Width: " + std::to_string(dftRtRect.size.width) + "Height: " + std::to_string(dftRtRect.size.height),
											cv::Point(dx + sx + defRoi.x, dy + sy + defRoi.y - 20 - 20 * c), 1, 1, cv::Scalar(0, 0, 255), 2);
									}

#endif // IMGSAVECUDA
									if (dftRtRect.size.width > widthMax)
									{
										widthMax = dftRtRect.size.width;
									}
									if (dftRtRect.size.height > heightMax)
									{
										heightMax = dftRtRect.size.height;
									}
									if (dftArea > areaMax)
									{
										areaMax = dftArea;
									}
								}
								if (areaMax > configParam.CarveParam.colorParam.infArea ||
									widthMax > configParam.CarveParam.colorParam.infWidth ||
									heightMax > configParam.CarveParam.colorParam.infHeight)

								{

									roi.x += dx;
									roi.y += dy;
									defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
									cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif
								}


							}
							break;

							case Layer::lineLay_finger:
							{
								if (defects[i].cfd < configParam.FingerParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif

							}
							break;

							case Layer::lineLay_baojiao:
							{
								if (defects[i].cfd < configParam.TransprencyParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif

							}
							break;

							case Layer::printLay_lvyou:
							{
								if (defects[i].cfd < configParam.TransprencyParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif

							}
							break;

							case Layer::lineLay_base:
							{
								if (defects[i].cfd < configParam.TransprencyParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif
							}
							break;

							case Layer::lineLay_nest:
							{
								if (defects[i].cfd < configParam.TransprencyParam.cfg)
								{
									continue;
								}

								roi.x += dx;
								roi.y += dy;
								defectInfo[defects[i].pcsId].dft.push_back({ gItems[n].iImgIndex, n, pAbstract->type, pItem->iID, roi });

#ifdef IMGSAVECUDA
								cv::rectangle(imgShow, { dx + sx + defRoi.x,dy + sy + defRoi.y,defRoi.width,defRoi.height }, cv::Scalar(0, 0, 255), 2);

#endif

							}
							break;

							default:
								break;
							}

							numDefectSelect++;
                        }
                        break;
                    }
                }

            }
        }
    }

#ifdef _TIME_LOG_
	std::chrono::steady_clock::time_point time5 = std::chrono::steady_clock::now();
	timeCostSingle = std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time4);
	timeCostFull = std::chrono::duration_cast<std::chrono::milliseconds>(time5 - timeStart);

	file <<"    "<< std::left << std::setw(32) <<strFunction + "5 分类筛选"
		<< std::right << std::setw(16) << timeCostFull.count()
		<< std::right << std::setw(16) << timeCostSingle.count()
	    << std::endl;
	file.close();

#endif

#ifdef IMGSAVECUDA
	//cv::imwrite(std::to_string(num) + "_Img_CUDA_A.jpg", imgShow);
	num++;
	log->info("--------CUDA 结果保存完毕--------");
#endif

	log->info("After Selected: " + std::to_string(numDefectSelect));
	
    return 0;
}

#ifdef IMGSAVECUDA
#undef  IMGSAVECUDA
#endif // IMGSAVECUDA


#endif

int AlgBase::padTrainer(
    const ABSTRACT_REGIONS & region,
    cv::Mat *img,
    PadParam *pads,
    Layer layer,
    int &nPad)
{
    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    nPad = -1;

    double	foreMean[3], foreStddev[3], foreArea[3],
        backMean[3], backStddev[3], backArea[3];

    double _area;

    uchar *pr, *pg, *pb, *pm;
    cv::Vec3b *pVec3b, *pImg;
    uchar *pUchar;
    cv::Mat channels[3];
    cv::Mat hsv;
    cv::Mat subImg;
    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        cv::morphologyEx(pItem->mask,
            mask,
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1,
            cv::BorderTypes::BORDER_CONSTANT,
            0);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        subImg = (*img)(roi).clone();

        //for (int r = 0; r < roi.height; r++)
        //{
        //    pVec3b = subImg.ptr<cv::Vec3b>(r);
        //    pUchar = mask.ptr<uchar>(r + maskRoi.y);
        //    pImg = img->ptr<cv::Vec3b>(r + roi.y);
        //    for (int c = 0; c < roi.width; c++)
        //    {
        //        if (pUchar[c + maskRoi.x] > 128)
        //        {
        //            pVec3b[c][0] = gLut.ptr(0)[pImg[c + roi.x][0]];
        //            pVec3b[c][1] = gLut.ptr(0)[pImg[c + roi.x][1]];
        //            pVec3b[c][2] = gLut.ptr(0)[pImg[c + roi.x][2]];
        //        }
        //    }
        //}

        //cv::cvtColor((*img)(roi), hsv, cv::COLOR_BGR2HSV);

        cv::split(subImg, channels);

        for (int c = 0; c < 3; c++)
        {
            classStddev(channels[c], mask(maskRoi),
                foreMean[c], foreStddev[c], foreArea[c],
                backMean[c], backStddev[c], backArea[c]);

            pads[pItem->iID].upperMean[c] = (pads[pItem->iID].upperMean[c] *
                pads[pItem->iID].upperArea[c] +
                foreMean[c] * foreArea[c]) /
                (foreArea[c] + pads[pItem->iID].upperArea[c] + 1e-7);

            pads[pItem->iID].upperStdDev[c] = (pads[pItem->iID].upperStdDev[c] *
                pads[pItem->iID].upperArea[c] +
                foreStddev[c] * foreArea[c]) /
                (foreArea[c] + pads[pItem->iID].upperArea[c] + 1e-7);

            pads[pItem->iID].upperArea[c] += foreArea[c];

            pads[pItem->iID].lowerMean[c] = (pads[pItem->iID].lowerMean[c] *
                pads[pItem->iID].lowerArea[c] +
                backMean[c] * backArea[c]) /
                (backArea[c] + pads[pItem->iID].lowerArea[c] + 1e-7);

            pads[pItem->iID].lowerStdDev[c] = (pads[pItem->iID].lowerStdDev[c] *
                pads[pItem->iID].lowerArea[c] +
                backStddev[c] * backArea[c]) /
                (backArea[c] + pads[pItem->iID].lowerArea[c] + 1e-7);

            pads[pItem->iID].lowerArea[c] += backArea[c];

            _area = cv::sum(mask(maskRoi)).val[0] / 255;

            pads[pItem->iID].totalMean[c] = (cv::mean(channels[c], mask(maskRoi)).val[0] * _area +
                pads[pItem->iID].totalMean[c] * pads[pItem->iID].totalArea[c]) /
                (_area + pads[pItem->iID].totalArea[c] + 1e-7);

            pads[pItem->iID].totalArea[c] += _area;

        }

        for (int r = 0; r < roi.height; r++)
        {
            pb = channels[0].ptr<uchar>(r);
            pg = channels[1].ptr<uchar>(r);
            pr = channels[2].ptr<uchar>(r);
            pm = mask(maskRoi).ptr<uchar>(r);
            for (int c = 0; c < roi.width; c++)
            {
                if (pm[c] < 128)	continue;
                pads[pItem->iID].hist[pb[c]]++;
                pads[pItem->iID].hist[pg[c] + 256]++;
                pads[pItem->iID].hist[pr[c] + 512]++;
            }
        }
        nPad = __max(nPad, pItem->iID);
    }
    nPad += 1;
    return 0;
}

int AlgBase::loadImageData(
	const ABSTRACT_REGIONS & region,
	cv::Mat * img,
	std::vector<std::vector<cv::Mat>> &vecImg,
	std::vector<std::vector<cv::Mat>> &vecMask,
	std::vector<unsigned int> &vecSumPix)
{
	const ITEM_REGION *pItem;
	cv::Mat mask;
	cv::Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);

	//items的起始结束区域
	int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
	cv::Rect roi, maskRoi;

	cv::Mat subImg;
	for (int i = 0; i < region.items.size(); i++)
	{
		pItem = &region.items[i];
		if (pItem->iID < 0)	continue;

		cv::morphologyEx(pItem->mask,
			mask,
			cv::MorphTypes::MORPH_ERODE,
			kernel,
			cv::Point(-1, -1),
			1,
			cv::BorderTypes::BORDER_CONSTANT,
			0);

		sx = __max(0, pItem->iOffsetX);
		sy = __max(0, pItem->iOffsetY);
		ex = __min(pItem->iOffsetX + mask.cols, img->cols);
		ey = __min(pItem->iOffsetY + mask.rows, img->rows);
		width = ex - sx;
		height = ey - sy;
		roi = { sx, sy, width, height };
		if (ex <= 0 || width <= 0)	continue;

		maskRoi = { 
			sx - pItem->iOffsetX,
			0,
			width,
			height };

		subImg = (*img)(roi).clone();

		vecImg[pItem->iID].push_back((*img)(roi).clone());
		vecMask[pItem->iID].push_back(mask(maskRoi));
		vecSumPix[pItem->iID] += cv::countNonZero(mask(maskRoi));
	}

	return 0;
}

int AlgBase::padTrainer_SGM(
	const ABSTRACT_REGIONS & region,
	cv::Mat * img,
	PadParam * pads,
	int & nPad,
	PadTrain_SGM *matrix)
{
	const ITEM_REGION *pItem;
	cv::Mat mask;
	cv::Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_8UC1);

	//items的起始结束区域
	int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
	cv::Rect roi, maskRoi;

	nPad = -1;
	cv::Mat _mean, _covar;
	cv::Mat subImg;
	uchar* data;
	cv::Mat _mask;
	cv::Mat matrixTemp;
	cv::Mat matrixValueFeature, matrixVectorFeature;
	cv::Vec3b *ptrImg;
	uchar *ptrMask;//掩膜操作
	for (int i = 0; i < region.items.size(); i++)
	{
		pItem = &region.items[i];
		if (pItem->iID < 0)	continue;

		cv::morphologyEx(pItem->mask,
			mask,
			cv::MorphTypes::MORPH_ERODE,
			kernel,
			cv::Point(-1, -1),
			1,
			cv::BorderTypes::BORDER_CONSTANT,
			0);

		sx = __max(0, pItem->iOffsetX);
		sy = __max(0, pItem->iOffsetY);
		ex = __min(pItem->iOffsetX + mask.cols, img->cols);
		ey = __min(pItem->iOffsetY + mask.rows, img->rows);
		width = ex - sx;
		height = ey - sy;
		roi = { sx, sy, width, height };
		if (ex <= 0 || width <= 0)	continue;

		maskRoi = {
			sx - pItem->iOffsetX,
			0,
			width,
			height };

		subImg = (*img)(roi).clone();
		 _mask = mask(maskRoi);

		cv::Mat imgShow;
		subImg.copyTo(imgShow,_mask);
		int sumNoZero = cv::countNonZero(_mask);

		//当前位置；
		int index = matrix[pItem->iID].rowMatrix;

		int count = 0;
		for (int r = 0; r < subImg.rows; r++)
		{
			ptrImg = subImg.ptr<cv::Vec3b>(r);
			ptrMask = _mask.ptr<uchar>(r);//掩膜操作
			for (int c = 0; c < subImg.cols; c++)
			{
				if (ptrMask[c] == 255)
				{
					
					matrix[pItem->iID].pMatrixTrain[index + count] = ptrImg[c][0];
					matrix[pItem->iID].pMatrixTrain[index + count +1] = ptrImg[c][1];
					matrix[pItem->iID].pMatrixTrain[index + count +2] = ptrImg[c][2];

					count = count + 3;
				}
			}
		}
		matrix[pItem->iID].rowMatrix += count;


		matrixTemp = cv::Mat((matrix[pItem->iID].rowMatrix)/3, 3, CV_8UC1, matrix[pItem->iID].pMatrixTrain);

		cv::calcCovarMatrix(matrixTemp, _covar, _mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);

		_covar /= (matrix[pItem->iID].rowMatrix / 3 - 1);

		std::cout << "Mean:" << _mean.at<double>(0, 0) << " " << _mean.at<double>(0, 1) << " "<< _mean.at<double>(0, 2) << "|" << std::endl;
		
		cv::eigen(_covar, matrixValueFeature, matrixVectorFeature);
		
		pads[pItem->iID].totalMean_SGM[0] = _mean.at<double>(0, 0);
		pads[pItem->iID].totalMean_SGM[1] = _mean.at<double>(0, 1);
		pads[pItem->iID].totalMean_SGM[2] = _mean.at<double>(0, 2);

		pads[pItem->iID].valueFeature[0] = matrixValueFeature.at<double>(0, 0);
		pads[pItem->iID].valueFeature[1] = matrixValueFeature.at<double>(1, 0);
		pads[pItem->iID].valueFeature[2] = matrixValueFeature.at<double>(2, 0);

		pads[pItem->iID].vectorFeature[0][0] = matrixVectorFeature.at<double>(0, 0);
		pads[pItem->iID].vectorFeature[0][1] = matrixVectorFeature.at<double>(0, 1);
		pads[pItem->iID].vectorFeature[0][2] = matrixVectorFeature.at<double>(0, 2);

		pads[pItem->iID].vectorFeature[1][0] = matrixVectorFeature.at<double>(1, 0);
		pads[pItem->iID].vectorFeature[1][1] = matrixVectorFeature.at<double>(1, 1);
		pads[pItem->iID].vectorFeature[1][2] = matrixVectorFeature.at<double>(1, 2);

		pads[pItem->iID].vectorFeature[2][0] = matrixVectorFeature.at<double>(2, 0);
		pads[pItem->iID].vectorFeature[2][1] = matrixVectorFeature.at<double>(2, 1);
		pads[pItem->iID].vectorFeature[2][2] = matrixVectorFeature.at<double>(2, 2);

		
		nPad = __max(nPad, pItem->iID);
		std::cout << pItem->iID << ": "<< matrix[pItem->iID].rowMatrix <<"||";
		//matrix[pItem->iID].rowMatrix += count;
		//std::cout << count << "-"<< sumNoZero*3 <<"End" << matrix[pItem->iID].rowMatrix << "|";
	}
	std::cout << std::endl;
	nPad += 1;
	return 0;
}

int AlgBase::steelTrainer(
    const ABSTRACT_REGIONS &region,
    cv::Mat *img,
    SteelParam *steel,
    Layer layer,
    int &nSteel)
{
    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel35 = cv::Mat::ones(cv::Size(35, 35), CV_8UC1);
    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    nSteel = -1;

    uchar *pr, *pg, *pb, *pm;

    double	foreMean[3], foreStddev[3], foreArea[3],
        backMean[3], backStddev[3], backArea[3];

    double _area;

    cv::Vec3b *pVec3b, *pImg;
    uchar *pUchar;
    cv::Mat channels[3];
    cv::Mat subImg;
    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        cv::morphologyEx(pItem->mask,
            mask,
            cv::MorphTypes::MORPH_ERODE,
            kernel35,
            cv::Point(-1, -1),
            1,
            cv::BorderTypes::BORDER_CONSTANT,
            0);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        subImg = (*img)(roi).clone();

        //for (int r = 0; r < roi.height; r++)
        //{
        //    pVec3b = subImg.ptr<cv::Vec3b>(r);
        //    pUchar = mask.ptr<uchar>(r + maskRoi.y);
        //    pImg = img->ptr<cv::Vec3b>(r + roi.y);
        //    for (int c = 0; c < roi.width; c++)
        //    {
        //        if (pUchar[c + maskRoi.x] > 128)
        //        {
        //            pVec3b[c][0] = gLut.ptr(0)[pImg[c + roi.x][0]];
        //            pVec3b[c][1] = gLut.ptr(0)[pImg[c + roi.x][1]];
        //            pVec3b[c][2] = gLut.ptr(0)[pImg[c + roi.x][2]];
        //        }
        //    }
        //}

        //cv::cvtColor((*img)(roi), hsv, cv::COLOR_BGR2HSV);

        cv::split(subImg, channels);

        for (int c = 0; c < 3; c++)
        {
            classStddev(channels[c], mask(maskRoi),
                foreMean[c], foreStddev[c], foreArea[c],
                backMean[c], backStddev[c], backArea[c]);

            steel[pItem->iID].upperMean[c] = (steel[pItem->iID].upperMean[c] *
                steel[pItem->iID].upperArea[c] +
                foreMean[c] * foreArea[c]) /
                (foreArea[c] + steel[pItem->iID].upperArea[c] + 1e-7);

            steel[pItem->iID].upperStdDev[c] = (steel[pItem->iID].upperStdDev[c] *
                steel[pItem->iID].upperArea[c] +
                foreStddev[c] * foreArea[c]) /
                (foreArea[c] + steel[pItem->iID].upperArea[c] + 1e-7);

            steel[pItem->iID].upperArea[c] += foreArea[c];

            steel[pItem->iID].lowerMean[c] = (steel[pItem->iID].lowerMean[c] *
                steel[pItem->iID].lowerArea[c] +
                backMean[c] * backArea[c]) /
                (backArea[c] + steel[pItem->iID].lowerArea[c] + 1e-7);

            steel[pItem->iID].lowerStdDev[c] = (steel[pItem->iID].lowerStdDev[c] *
                steel[pItem->iID].lowerArea[c] +
                backStddev[c] * backArea[c]) /
                (backArea[c] + steel[pItem->iID].lowerArea[c] + 1e-7);

            steel[pItem->iID].lowerArea[c] += backArea[c];

            _area = cv::sum(mask(maskRoi)).val[0] / 255;

            steel[pItem->iID].totalMean[c] = (cv::mean(channels[c], mask(maskRoi)).val[0] * _area +
                steel[pItem->iID].totalMean[c] * steel[pItem->iID].totalArea[c]) /
                (_area + steel[pItem->iID].totalArea[c] + 1e-7);

            steel[pItem->iID].totalArea[c] += _area;
        }

        for (int r = 0; r < roi.height; r++)
        {
            pb = channels[0].ptr<uchar>(r);
            pg = channels[1].ptr<uchar>(r);
            pr = channels[2].ptr<uchar>(r);
            pm = mask(maskRoi).ptr<uchar>(r);
            for (int c = 0; c < roi.width; c++)
            {
                if (pm[c] < 128)	continue;
                steel[pItem->iID].hist[pb[c]]++;
                steel[pItem->iID].hist[pg[c] + 256]++;
                steel[pItem->iID].hist[pr[c] + 512]++;
            }
        }

        nSteel = __max(nSteel, pItem->iID);
    }
    nSteel += 1;
    return 0;
}

int AlgBase::opacityTrainer(
    const ABSTRACT_REGIONS &region,
    cv::Mat *img,
    OpacityParam *opacity,
    Layer layer,
    int &nOpacity)
{
    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel35 = cv::Mat::ones(cv::Size(35, 35), CV_8UC1);
    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    nOpacity = -1;

    uchar *pr, *pg, *pb, *pm;

    double	foreMean[3], foreStddev[3], foreArea[3],
        backMean[3], backStddev[3], backArea[3];

    double _area;

    cv::Vec3b *pVec3b;
    uchar *pUchar;
    cv::Mat channels[3];
    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        cv::morphologyEx(pItem->mask,
            mask,
            cv::MorphTypes::MORPH_ERODE,
            kernel35,
            cv::Point(-1, -1),
            1,
            cv::BorderTypes::BORDER_CONSTANT,
            0);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        cv::split((*img)(roi), channels);
        for (int c = 0; c < 3; c++)
        {
            classStddev(channels[c], mask(maskRoi),
                foreMean[c], foreStddev[c], foreArea[c],
                backMean[c], backStddev[c], backArea[c]);

            opacity[pItem->iID].upperMean[c] = (opacity[pItem->iID].upperMean[c] *
                opacity[pItem->iID].upperArea[c] +
                foreMean[c] * foreArea[c]) /
                (foreArea[c] + opacity[pItem->iID].upperArea[c] + 1e-7);

            opacity[pItem->iID].upperStdDev[c] = (opacity[pItem->iID].upperStdDev[c] *
                opacity[pItem->iID].upperArea[c] +
                foreStddev[c] * foreArea[c]) /
                (foreArea[c] + opacity[pItem->iID].upperArea[c] + 1e-7);

            opacity[pItem->iID].upperArea[c] += foreArea[c];

            opacity[pItem->iID].lowerMean[c] = (opacity[pItem->iID].lowerMean[c] *
                opacity[pItem->iID].lowerArea[c] +
                backMean[c] * backArea[c]) /
                (backArea[c] + opacity[pItem->iID].lowerArea[c] + 1e-7);

            opacity[pItem->iID].lowerStdDev[c] = (opacity[pItem->iID].lowerStdDev[c] *
                opacity[pItem->iID].lowerArea[c] +
                backStddev[c] * backArea[c]) /
                (backArea[c] + opacity[pItem->iID].lowerArea[c] + 1e-7);

            opacity[pItem->iID].lowerArea[c] += backArea[c];


            _area = cv::sum(mask(maskRoi)).val[0] / 255;

            opacity[pItem->iID].totalMean[c] = (cv::mean(channels[c], mask(maskRoi)).val[0] * _area +
                opacity[pItem->iID].totalMean[c] * opacity[pItem->iID].totalArea[c]) /
                (_area + opacity[pItem->iID].totalArea[c] + 1e-7);

            opacity[pItem->iID].totalArea[c] += _area;
        }

        for (int r = 0; r < roi.height; r++)
        {
            pb = channels[0].ptr<uchar>(r);
            pg = channels[1].ptr<uchar>(r);
            pr = channels[2].ptr<uchar>(r);
            pm = mask(maskRoi).ptr<uchar>(r);
            for (int c = 0; c < roi.width; c++)
            {
                if (pm[c] < 128)	continue;
                opacity[pItem->iID].hist[pb[c]]++;
                opacity[pItem->iID].hist[pg[c] + 256]++;
                opacity[pItem->iID].hist[pr[c] + 512]++;
            }
        }

        nOpacity = __max(nOpacity, pItem->iID);
    }
    nOpacity += 1;
    return 0;
}

int AlgBase::transprencyTrainer(
    const ABSTRACT_REGIONS &region,
    cv::Mat *img,
    TransparencyParam *transparency,
    Layer layer,
    int pcsId,
    int &nTransprency)
{

	static int num = 0;

    const ITEM_REGION *pItem;
    cv::Mat mask;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    uchar *pm;
    cv::Vec3s *prgbs;

    nTransprency = -1;

    cv::Vec3b *pVec3b;
    uchar *pUchar;

    int detectSize = 5;

    cv::Mat channels[3];
    cv::Mat subImg;
    cv::Mat reSizeImg = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
    cv::Mat blur16s, img16s;
    cv::Mat kernel = cv::Mat::ones(detectSize, detectSize, CV_8UC1);

    int _pos, _num;
    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        mask = pItem->mask.clone();
        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);

        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        cv::morphologyEx(mask(maskRoi),
            mask(maskRoi),
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1,
            cv::BorderTypes::BORDER_CONSTANT,
            0);

        subImg = (*img)(roi);

		double area = cv::sum(mask(maskRoi)).val[0] / 255;

		if (area < 100)
		{
			continue;
		}
        subImg.convertTo(img16s, CV_16SC3);
        cv::GaussianBlur(img16s, blur16s, cv::Size(detectSize, detectSize), 2);
        img16s = img16s - blur16s;

		/*cv::imwrite("data\\nest\\"+std::to_string(pItem->iID)+"_"+ std::to_string(num)+"_s.jpg", subImg);
		cv::imwrite("data\\nest\\" + std::to_string(pItem->iID) + "_" + std::to_string(num) + "_m.jpg", mask(maskRoi));
		cv::Mat imgResult;
		subImg.copyTo(imgResult, mask(maskRoi));
		cv::imwrite("data\\nest\\"+std::to_string(pItem->iID)+"_"+ std::to_string(num)+"_r.jpg", imgResult);*/

        //差异中心转为255，减少判断计算
        for (int r = 0; r < roi.height; r++)
        {
            pVec3b = subImg.ptr<cv::Vec3b>(r);
            pm = mask(maskRoi).ptr<uchar>(r);
            prgbs = img16s.ptr<cv::Vec3s>(r);
            for (int c = 0; c < roi.width; c++)
            {
                if (pm[c] > 128)
                {
                    transparency[pItem->iID].histColor[pVec3b[c][0]]++;
                    transparency[pItem->iID].histColor[pVec3b[c][1] + 256]++;
                    transparency[pItem->iID].histColor[pVec3b[c][2] + 512]++;
                    transparency[pItem->iID].histDiff[prgbs[c][0] + 256]++;
                    transparency[pItem->iID].histDiff[prgbs[c][1] + 768]++;
                    transparency[pItem->iID].histDiff[prgbs[c][2] + 1280]++;
                    transparency[pItem->iID].totalArea++;
                }
            }
        }

        for (int c = 0; c < 3; c++)
        {
            _pos = 0;
            _num = transparency[pItem->iID].totalNum;
            //暗分量
            extractMode(transparency[pItem->iID].histDiff, c * 2 * 256 + 1, (c * 2 + 1) * 256 - 1, _pos);
            _pos -= 256 * (c * 2 + 1);
            transparency[pItem->iID].lower[c] = (transparency[pItem->iID].lower[c] * _num + _pos) / (_num + 1);
            //亮分量
            extractMode(transparency[pItem->iID].histDiff, (c * 2 + 1) * 256 + 1, (c * 2 + 2) * 256 - 1, _pos);
            _pos -= 256 * (c * 2 + 1);
            transparency[pItem->iID].upper[c] = (transparency[pItem->iID].upper[c] * _num + _pos) / (_num + 1);
        }
        transparency[pItem->iID].totalNum++;

        nTransprency = __max(nTransprency, pItem->iID);
    }
    nTransprency++;
	num++;
    return 0;
}

int AlgBase::transprencyTrainerV2(
    const ABSTRACT_REGIONS & region,
    cv::Mat * img,
    TransparencyParam * transparency,
    Layer layer,
    int pcsId,
    int & nTransprency)
{
    return 0;
}

int AlgBase::lineTrainer(
    const ABSTRACT_REGIONS &region,
    cv::Mat *img,
    LineParam *line)
{
    return 0;
}

int AlgBase::fingerTrainer(
    const ABSTRACT_REGIONS &region,
    cv::Mat *img,
    FingerParam *finger)
{
    return 0;
}

int AlgBase::holeTrainer(
    const ABSTRACT_REGIONS &region,
    cv::Mat * img,
    HoleParam * hole,
    int & nHole)
{
    const ITEM_REGION *pItem;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    uchar *pr, *pg, *pb, *pm;

    nHole = -1;

    cv::Vec3b *pVec3b;
    uchar *pUchar;

    int detectSize = 5;

    cv::Mat channels[3];
    cv::Mat subImg;
    cv::Mat reSizeImg = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        sx = __max(0, pItem->iOffsetX - pItem->mask.cols);
        sy = __max(0, pItem->iOffsetY - pItem->mask.rows);
        ex = __min(pItem->iOffsetX + pItem->mask.cols * 1.5, img->cols);
        ey = __min(pItem->iOffsetY + pItem->mask.rows * 1.5, img->rows);

        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        subImg = (*img)(roi);
        cv::split(subImg, channels);

        double maxEntropy = 0;
        //提取当前最大熵的分量
        for (int c = 0; c < 3; c++)
        {
            cv::resize(channels[c], reSizeImg, reSizeImg.size());
            double etp = entropy(reSizeImg);
            if (maxEntropy < etp)
            {
                maxEntropy = etp;
                hole[pItem->iID].chioseChannel = c;
            }
        }
        nHole = __max(nHole, pItem->iID);
    }
    nHole++;
    return 0;
}

int AlgBase::carveTrainer(
    const ABSTRACT_REGIONS & region,
    cv::Mat * img,
    CarveParam * carve,
    Layer layer,
    int & nCarve)
{
    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel35 = cv::Mat::ones(cv::Size(3, 3), CV_8UC1);
    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    nCarve = -1;

    uchar *pr, *pg, *pb, *pm;

    double	foreMean[3], foreStddev[3], foreArea[3],
        backMean[3], backStddev[3], backArea[3];

    double _area;

    cv::Vec3b *pVec3b;
    uchar *pUchar;
    cv::Mat channels[3];
    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        cv::morphologyEx(pItem->mask,
            mask,
            cv::MorphTypes::MORPH_ERODE,
            kernel35,
            cv::Point(-1, -1),
            1,
            cv::BorderTypes::BORDER_CONSTANT,
            0);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        cv::split((*img)(roi), channels);
        for (int c = 0; c < 3; c++)
        {
            classStddev(channels[c], mask(maskRoi),
                foreMean[c], foreStddev[c], foreArea[c],
                backMean[c], backStddev[c], backArea[c]);

            carve[pItem->iID].upperMean[c] = (carve[pItem->iID].upperMean[c] *
                carve[pItem->iID].upperArea[c] +
                foreMean[c] * foreArea[c]) /
                (foreArea[c] + carve[pItem->iID].upperArea[c] + 1e-7);

            carve[pItem->iID].upperStdDev[c] = (carve[pItem->iID].upperStdDev[c] *
                carve[pItem->iID].upperArea[c] +
                foreStddev[c] * foreArea[c]) /
                (foreArea[c] + carve[pItem->iID].upperArea[c] + 1e-7);

            carve[pItem->iID].upperArea[c] += foreArea[c];

            carve[pItem->iID].lowerMean[c] = (carve[pItem->iID].lowerMean[c] *
                carve[pItem->iID].lowerArea[c] +
                backMean[c] * backArea[c]) /
                (backArea[c] + carve[pItem->iID].lowerArea[c] + 1e-7);

            carve[pItem->iID].lowerStdDev[c] = (carve[pItem->iID].lowerStdDev[c] *
                carve[pItem->iID].lowerArea[c] +
                backStddev[c] * backArea[c]) /
                (backArea[c] + carve[pItem->iID].lowerArea[c] + 1e-7);

            carve[pItem->iID].lowerArea[c] += backArea[c];


            _area = cv::sum(mask(maskRoi)).val[0] / 255;

            carve[pItem->iID].totalMean[c] = (cv::mean(channels[c], mask(maskRoi)).val[0] * _area +
                carve[pItem->iID].totalMean[c] * carve[pItem->iID].totalArea[c]) /
                (_area + carve[pItem->iID].totalArea[c] + 1e-7);

            carve[pItem->iID].totalArea[c] += _area;
        }

        for (int r = 0; r < roi.height; r++)
        {
            pb = channels[0].ptr<uchar>(r);
            pg = channels[1].ptr<uchar>(r);
            pr = channels[2].ptr<uchar>(r);
            pm = mask(maskRoi).ptr<uchar>(r);
            for (int c = 0; c < roi.width; c++)
            {
                if (pm[c] < 128)	continue;
                carve[pItem->iID].hist[pb[c]]++;
                carve[pItem->iID].hist[pg[c] + 256]++;
                carve[pItem->iID].hist[pr[c] + 512]++;
            }
        }

        nCarve = __max(nCarve, pItem->iID);
    }
    nCarve += 1;
    return 0;
    return 0;
}

int AlgBase::simpleTrain(
    const ABSTRACT_REGIONS & region, 
    const cv::Mat &img,
    TrainParam & param, 
    Layer layer)
{
    const ITEM_REGION *pItem;

    cv::Mat mask;

    int knlSize = 2;
    
    switch (layer)
    {
    case lineLay_pad:
        knlSize = 2;
        break;
    case stiffenerLay_steel:
        knlSize = 10;
        break;
    case printLay_EMI:
        knlSize = 10;
        break;
    case carveLay:
        knlSize = 10;
        break;
    default:
        break;
    }
    
    cv::Mat kernel = cv::Mat::ones(cv::Size(knlSize * 2 + 1, knlSize * 2 + 1), CV_8UC1);

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    cv::Scalar mean, stddev;

    cv::Mat subImg;

    int itemIndex = param.getSimpleIndex(layer);

    if (itemIndex == -1)
    {
        itemIndex = param.nSimple;                 //新layer更新

        param.nSimple++;
    }

    if (itemIndex >= TRAIN_SPACE)   return -1;

    SimpleParam *pSimpleParam = &param.simpleParam[itemIndex];

    pSimpleParam->layer = layer;

    MeanStd *pMeanStd = NULL;

    float _area;
    
    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        pMeanStd = pSimpleParam->data + pItem->iID;

        cv::threshold(pItem->mask, mask, 128, 255, CV_THRESH_BINARY);

        cv::morphologyEx(pItem->mask,
                        mask,
                        cv::MorphTypes::MORPH_ERODE,
                        kernel,
                        cv::Point(-1, -1),
                        1,
                        cv::BorderTypes::BORDER_CONSTANT,
                        0);

        sx = __max(0, pItem->iOffsetX);

        sy = __max(0, pItem->iOffsetY);

        ex = __min(pItem->iOffsetX + mask.cols, img.cols);

        ey = __min(pItem->iOffsetY + mask.rows, img.rows);

        width = ex - sx;

        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
                    0,
                    width,
                    height };

                subImg = img(roi).clone();

        cv::meanStdDev(subImg, mean, stddev, mask(maskRoi));

        _area = (float)cv::sum(mask(maskRoi)).val[0] / PIXEL_NORMALIZATION;

        for (int c = 0; c < 3; c++)
        {
            pMeanStd->mean[c] = (pMeanStd->mean[c] * pMeanStd->area + mean.val[c] / PIXEL_NORMALIZATION * _area ) / (pMeanStd->area + _area);

            pMeanStd->stddev[c] = std::sqrt((std::pow(pMeanStd->stddev[c],2) * pMeanStd->area + std::pow((stddev.val[c] / PIXEL_NORMALIZATION),2) * _area ) / (pMeanStd->area + _area));
        }
        
        pMeanStd->area += _area;

        pSimpleParam->nItem = __max(pItem->iID + 1, pSimpleParam->nItem);
    }
    return 0;
}

int AlgBase::complexTrain(
    const ABSTRACT_REGIONS & region, 
    const cv::Mat & img, 
    TrainParam & param, 
    Layer layer)
{
    return 0;
}

int AlgBase::simpleInspector(
    const ABSTRACT_REGIONS &region, 
    const cv::Mat &img, 
    const ConfigParam &configParam,
    TrainParam &param, 
    Layer layer,
    std::vector<DFTINFO>& defectInfo)
{
    const ITEM_REGION *pItem;

    //分配内存
    int nMem = 0;

    uchar *knSpc = NULL;

    float *pcsSpc = NULL;

    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);

        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);

        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }
    
    int shrinkSize = 5;
    
    uchar tLower[3] = {0}, tUpper[3] = {255,255,255};

    //像素权重加成
    double areaWeight = 1;  //钢片尝试不用

    //面积、高度、宽度限制
    float infArea = 0, infHeight = 0, infWidth = 0;

    //融合轴距离
    int fuseDist = 0;

    switch (layer)
    {
    case lineLay_pad:
        shrinkSize = configParam.PadParam.shrinkSize * 2 + 1;

        infArea = configParam.PadParam.colorParam.infArea;

        infWidth = configParam.PadParam.colorParam.infWidth;

        infHeight = configParam.PadParam.colorParam.infHeight;

        fuseDist = configParam.PadParam.ruseDist;

        for (int i = 0; i < 3; i++)
        {
            tLower[i] = configParam.PadParam.colorParam.lowerLimit[i];
            tUpper[i] = configParam.PadParam.colorParam.upperLimit[i];
        }

        break;

    case stiffenerLay_steel:
        shrinkSize = configParam.SteelParam.shrinkSize * 2 + 1;

        infArea = configParam.SteelParam.colorParam.infArea;

        infWidth = configParam.SteelParam.colorParam.infWidth;

        infHeight = configParam.SteelParam.colorParam.infHeight;

        fuseDist = configParam.SteelParam.ruseDist;

        for (int i = 0; i < 3; i++)
        {
            tLower[i] = configParam.SteelParam.colorParam.lowerLimit[i];
            tUpper[i] = configParam.SteelParam.colorParam.upperLimit[i];
        }

        break;

    case printLay_EMI:
        shrinkSize = configParam.OpacityParam.shrinkSize * 2 + 1;

        infArea = configParam.OpacityParam.colorParam.infArea;

        infWidth = configParam.OpacityParam.colorParam.infWidth;

        infHeight = configParam.OpacityParam.colorParam.infHeight;

        fuseDist = configParam.OpacityParam.ruseDist;

        for (int i = 0; i < 3; i++)
        {
            tLower[i] = configParam.OpacityParam.colorParam.lowerLimit[i];
            tUpper[i] = configParam.OpacityParam.colorParam.upperLimit[i];
        }

        break;

    case carveLay:
        shrinkSize = configParam.CarveParam.shrinkSize * 2 + 1;

        infArea = configParam.CarveParam.colorParam.infArea;

        infWidth = configParam.CarveParam.colorParam.infWidth;

        infHeight = configParam.CarveParam.colorParam.infHeight;

        fuseDist = configParam.CarveParam.ruseDist;

        for (int i = 0; i < 3; i++)
        {
            tLower[i] = configParam.CarveParam.colorParam.lowerLimit[i];
            tUpper[i] = configParam.CarveParam.colorParam.upperLimit[i];
        }

        break;

    default:
        break;
    }

    cv::Mat mask;

    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);

    cv::Mat dst;

    cv::Mat mLower, mUpper;

    cv::Rect dftRect;

    std::vector<cv::Rect> dftRects;

    cv::RotatedRect dftRtRect;

    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    bool totalFlag = false;

    cv::Scalar mean;

    cv::Mat channelMat[3];

    cv::Mat subImg;

    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        
        cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);

        sy = __max(0, pItem->iOffsetY);

        ex = __min(pItem->iOffsetX + mask.cols, img.cols);

        ey = __min(pItem->iOffsetY + mask.rows, img.rows);

        width = ex - sx;

        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX, 0, width, height};

        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);

        channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);

        channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);

        channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);

        subImg = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);

        dst.setTo(0);

        subImg = (img)(roi).clone();

        cv::split(subImg, channelMat);

        for (int c = 0; c < 3; c++)
        {
            cv::threshold(channelMat[c], mLower, tLower[c], 255, cv::THRESH_BINARY_INV);

            cv::threshold(channelMat[c], mUpper, tUpper[c], 255, cv::THRESH_BINARY);

            dst = dst + mLower + mUpper;
        }

        //maskFitting(~dst, mask(maskRoi));
        cv::morphologyEx(mask(maskRoi),
                        mask(maskRoi),
                        cv::MorphTypes::MORPH_ERODE,
                        kernel,
                        cv::Point(-1, -1),
                        1, cv::BorderTypes::BORDER_CONSTANT, 0);

        dst = dst & mask(maskRoi);

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;

            if (dftArea > infArea ||
                dftRtRect.size.width > infWidth ||
                dftRtRect.size.height > infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back({dftRects[n], pItem->iID});
            }
        }

        dftRects.clear();
    }

    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::padInspectorV2(
    const ABSTRACT_REGIONS & region, 
    const cv::Mat & img, 
    const ConfigParam & configParam, 
    TrainParam & param, 
    Layer layer, 
    std::vector<DFTINFO>& defectInfo)
{
    const ITEM_REGION *pItem;

    //寻找Layer对应的训练参数
    int layerIdx = param.getSimpleIndex(layer);

    if (layerIdx == -1) return -1;

    //分配内存
    int nMem = 0;

    uchar *knSpc = NULL;

    float *pcsSpc = NULL;

    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);

        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);

        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = 5;

    uchar tLower[3] = { 0 }, tUpper[3] = { 255,255,255 };

    //像素权重加成
    double areaWeight = 1;  //钢片尝试不用
    
    //面积、高度、宽度限制
    float infArea = 0, infHeight = 0, infWidth = 0;

    //融合轴距离
    int fuseDist = 0;

    shrinkSize = configParam.PadParam.shrinkSize * 2 + 1;

    infArea = configParam.PadParam.colorParam.infArea;

    infWidth = configParam.PadParam.colorParam.infWidth;

    infHeight = configParam.PadParam.colorParam.infHeight;

    fuseDist = configParam.PadParam.ruseDist;
 
    cv::Mat mask;

    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);

    cv::Mat dst;

    cv::Mat mLower, mUpper;

    cv::Rect dftRect;

    std::vector<cv::Rect> dftRects;

    cv::RotatedRect dftRtRect;

    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    bool totalFlag = false;

    cv::Scalar mean;

    cv::Mat channelMat[3];

    cv::Mat subImg;

    MeanStd *pMeanStd = NULL;

    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);

        sy = __max(0, pItem->iOffsetY);

        ex = __min(pItem->iOffsetX + mask.cols, img.cols);

        ey = __min(pItem->iOffsetY + mask.rows, img.rows);

        width = ex - sx;

        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX, 0, width, height };

        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);

        channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);

        channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);

        channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);

        subImg = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);

        dst.setTo(0);

        subImg = (img)(roi).clone();

        cv::split(subImg, channelMat);

        pMeanStd = &param.simpleParam[layerIdx].data[pItem->iID];

        //是否采用统一检测参数
        if(pMeanStd->unify)
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = configParam.PadParam.colorParam.lowerLimit[i];
                tUpper[i] = configParam.PadParam.colorParam.upperLimit[i];
            }
        }
        else
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = __max(0, pMeanStd->mean[i] - pMeanStd->stddev[i] * pMeanStd->lower[i]);  //configParam.PadParam.colorParam.lowerLimit[i];
                tUpper[i] = __min(255, pMeanStd->mean[i] + pMeanStd->stddev[i] * pMeanStd->upper[i]); //configParam.PadParam.colorParam.upperLimit[i];
            }
        }

        for (int c = 0; c < 3; c++)
        {
            cv::threshold(channelMat[c], mLower, tLower[c] - 1, 255, cv::THRESH_BINARY_INV);

            cv::threshold(channelMat[c], mUpper, tUpper[c], 255, cv::THRESH_BINARY);

            dst = dst + mLower + mUpper;
        }

        maskFitting(~dst, mask(maskRoi));
        cv::morphologyEx(mask(maskRoi),
            mask(maskRoi),
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);

        dst = dst & mask(maskRoi);

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;

            if (dftArea > infArea ||
                dftRtRect.size.width > infWidth ||
                dftRtRect.size.height > infHeight)
            {

                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back({ dftRects[n], pItem->iID });
            }
        }

        dftRects.clear();
    }

    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::steelInspectorV2(
    const ABSTRACT_REGIONS & region, 
    const cv::Mat & img, 
    const ConfigParam & configParam, 
    TrainParam & param, 
    Layer layer, 
    std::vector<DFTINFO>& defectInfo)
{
    const ITEM_REGION *pItem;

    //寻找Layer对应的训练参数
    int layerIdx = param.getSimpleIndex(layer);

    if (layerIdx == -1) return -1;

    //分配内存
    int nMem = 0;

    uchar *knSpc = NULL;

    float *pcsSpc = NULL;

    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);

        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);

        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = 5;

    uchar tLower[3] = { 0 }, tUpper[3] = { 255,255,255 };

    //像素权重加成
    double areaWeight = 1;  //钢片尝试不用

                            //面积、高度、宽度限制
    float infArea = 0, infHeight = 0, infWidth = 0;

    //融合轴距离
    int fuseDist = 0;

    shrinkSize = configParam.SteelParam.shrinkSize * 2 + 1;

    infArea = configParam.SteelParam.colorParam.infArea;

    infWidth = configParam.SteelParam.colorParam.infWidth;

    infHeight = configParam.SteelParam.colorParam.infHeight;

    fuseDist = configParam.SteelParam.ruseDist;

    cv::Mat mask;

    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);

    cv::Mat dst;

    cv::Mat mLower, mUpper;

    cv::Rect dftRect;

    std::vector<cv::Rect> dftRects;

    cv::RotatedRect dftRtRect;

    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    bool totalFlag = false;

    cv::Scalar mean;

    cv::Mat channelMat[3];

    cv::Mat subImg;

    MeanStd *pMeanStd = NULL;

    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);

        sy = __max(0, pItem->iOffsetY);

        ex = __min(pItem->iOffsetX + mask.cols, img.cols);

        ey = __min(pItem->iOffsetY + mask.rows, img.rows);

        width = ex - sx;

        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX, 0, width, height };

        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);

        channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);

        channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);

        channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);

        subImg = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);

        dst.setTo(0);

        subImg = (img)(roi).clone();

        cv::split(subImg, channelMat);

        pMeanStd = &param.simpleParam[layerIdx].data[pItem->iID];

        //是否采用统一检测参数
        if (pMeanStd->unify)
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = configParam.SteelParam.colorParam.lowerLimit[i];
                tUpper[i] = configParam.SteelParam.colorParam.upperLimit[i];
            }
        }
        else
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = __max(0, pMeanStd->mean[i] - pMeanStd->stddev[i] * pMeanStd->lower[i]);
                tUpper[i] = __min(255, pMeanStd->mean[i] + pMeanStd->stddev[i] * pMeanStd->upper[i]);
            }
        }

        for (int c = 0; c < 3; c++)
        {
            cv::threshold(channelMat[c], mLower, tLower[c] - 1, 255, cv::THRESH_BINARY_INV);

            cv::threshold(channelMat[c], mUpper, tUpper[c], 255, cv::THRESH_BINARY);

            dst = dst + mLower + mUpper;
        }

        //maskFitting(~dst, mask(maskRoi));
        cv::morphologyEx(mask(maskRoi),
            mask(maskRoi),
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);

        dst = dst & mask(maskRoi);

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;

            if (dftArea > infArea ||
                dftRtRect.size.width > infWidth ||
                dftRtRect.size.height > infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back({ dftRects[n], pItem->iID });
            }
        }

        dftRects.clear();
    }

    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::emiInspectorV2(
    const ABSTRACT_REGIONS & region, 
    const cv::Mat & img,    
    const ConfigParam & configParam,   
    TrainParam & param, 
    Layer layer, 
    std::vector<DFTINFO>& defectInfo)
{
    const ITEM_REGION *pItem;

    //寻找Layer对应的训练参数
    int layerIdx = param.getSimpleIndex(layer);

    if (layerIdx == -1) return -1;

    //分配内存
    int nMem = 0;

    uchar *knSpc = NULL;

    float *pcsSpc = NULL;

    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);

        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);

        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = 5;

    uchar tLower[3] = { 0 }, tUpper[3] = { 255,255,255 };

    //像素权重加成
    double areaWeight = 1;  //钢片尝试不用

                            //面积、高度、宽度限制
    float infArea = 0, infHeight = 0, infWidth = 0;

    //融合轴距离
    int fuseDist = 0;

    shrinkSize = configParam.OpacityParam.shrinkSize * 2 + 1;

    infArea = configParam.OpacityParam.colorParam.infArea;

    infWidth = configParam.OpacityParam.colorParam.infWidth;

    infHeight = configParam.OpacityParam.colorParam.infHeight;

    fuseDist = configParam.OpacityParam.ruseDist;

    cv::Mat mask;

    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);

    cv::Mat dst;

    cv::Mat mLower, mUpper;

    cv::Rect dftRect;

    std::vector<cv::Rect> dftRects;

    cv::RotatedRect dftRtRect;

    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    bool totalFlag = false;

    cv::Scalar mean;

    cv::Mat channelMat[3];

    cv::Mat subImg;

    MeanStd *pMeanStd = NULL;

    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);

        sy = __max(0, pItem->iOffsetY);

        ex = __min(pItem->iOffsetX + mask.cols, img.cols);

        ey = __min(pItem->iOffsetY + mask.rows, img.rows);

        width = ex - sx;

        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX, 0, width, height };

        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);

        channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);

        channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);

        channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);

        subImg = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);

        dst.setTo(0);

        subImg = (img)(roi).clone();

        cv::split(subImg, channelMat);

        pMeanStd = &param.simpleParam[layerIdx].data[pItem->iID];

        //是否采用统一检测参数
        if (pMeanStd->unify)
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = configParam.OpacityParam.colorParam.lowerLimit[i];
                tUpper[i] = configParam.OpacityParam.colorParam.upperLimit[i];
            }
        }
        else
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = __max(0, pMeanStd->mean[i] - pMeanStd->stddev[i] * pMeanStd->lower[i]);
                tUpper[i] = __min(255, pMeanStd->mean[i] + pMeanStd->stddev[i] * pMeanStd->upper[i]);
            }
        }

        for (int c = 0; c < 3; c++)
        {
            cv::threshold(channelMat[c], mLower, tLower[c] - 1, 255, cv::THRESH_BINARY_INV);

            cv::threshold(channelMat[c], mUpper, tUpper[c], 255, cv::THRESH_BINARY);

            dst = dst + mLower + mUpper;
        }

        //maskFitting(~dst, mask(maskRoi));
        cv::morphologyEx(mask(maskRoi),
            mask(maskRoi),
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);

        dst = dst & mask(maskRoi);

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;

            if (dftArea > infArea ||
                dftRtRect.size.width > infWidth ||
                dftRtRect.size.height > infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back({ dftRects[n], pItem->iID });
            }
        }

        dftRects.clear();
    }

    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::carveInspectorV2(
    const ABSTRACT_REGIONS & region, 
    const cv::Mat & img, 
    const ConfigParam & configParam, 
    TrainParam & param, Layer layer, 
    std::vector<DFTINFO>& defectInfo)
{
    const ITEM_REGION *pItem;

    //寻找Layer对应的训练参数
    int layerIdx = param.getSimpleIndex(layer);

    if (layerIdx == -1) return -1;

    //分配内存
    int nMem = 0;

    uchar *knSpc = NULL;

    float *pcsSpc = NULL;

    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);

        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);

        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = 5;

    uchar tLower[3] = { 0 }, tUpper[3] = { 255,255,255 };

    //像素权重加成
    double areaWeight = 1;  //钢片尝试不用

                            //面积、高度、宽度限制
    float infArea = 0, infHeight = 0, infWidth = 0;

    //融合轴距离
    int fuseDist = 0;

    shrinkSize = configParam.CarveParam.shrinkSize * 2 + 1;

    infArea = configParam.CarveParam.colorParam.infArea;

    infWidth = configParam.CarveParam.colorParam.infWidth;

    infHeight = configParam.CarveParam.colorParam.infHeight;

    fuseDist = configParam.CarveParam.ruseDist;

    cv::Mat mask;

    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);

    cv::Mat dst;

    cv::Mat mLower, mUpper;

    cv::Rect dftRect;

    std::vector<cv::Rect> dftRects;

    cv::RotatedRect dftRtRect;

    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    bool totalFlag = false;

    cv::Scalar mean;

    cv::Mat channelMat[3];

    cv::Mat subImg;

    MeanStd *pMeanStd = NULL;

    for (int i = 0; i < region.items.size(); i++)
    {
        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);

        sy = __max(0, pItem->iOffsetY);

        ex = __min(pItem->iOffsetX + mask.cols, img.cols);

        ey = __min(pItem->iOffsetY + mask.rows, img.rows);

        width = ex - sx;

        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX, 0, width, height };

        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);

        channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);

        channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);

        channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);

        subImg = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);

        dst.setTo(0);

        subImg = (img)(roi).clone();

        cv::split(subImg, channelMat);

        pMeanStd = &param.simpleParam[layerIdx].data[pItem->iID];

        //是否采用统一检测参数
        if (pMeanStd->unify)
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = configParam.CarveParam.colorParam.lowerLimit[i];
                tUpper[i] = configParam.CarveParam.colorParam.upperLimit[i];
            }
        }
        else
        {
            for (int i = 0; i < 3; i++)
            {
                tLower[i] = __max(0, pMeanStd->mean[i] - pMeanStd->stddev[i] * pMeanStd->lower[i]);
                tUpper[i] = __min(255, pMeanStd->mean[i] + pMeanStd->stddev[i] * pMeanStd->upper[i]);
            }
        }

        for (int c = 0; c < 3; c++)
        {
            cv::threshold(channelMat[c], mLower, tLower[c] - 1, 255, cv::THRESH_BINARY_INV);

            cv::threshold(channelMat[c], mUpper, tUpper[c], 255, cv::THRESH_BINARY);

            dst = dst + mLower + mUpper;
        }

        //maskFitting(~dst, mask(maskRoi));
        cv::morphologyEx(mask(maskRoi),
            mask(maskRoi),
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);

        dst = dst & mask(maskRoi);

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;

            if (dftArea > infArea ||
                dftRtRect.size.width > infWidth ||
                dftRtRect.size.height > infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back({ dftRects[n], pItem->iID });
            }
        }

        dftRects.clear();
    }

    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::complexInspector(
    const ABSTRACT_REGIONS & region, 
    const cv::Mat & img, 
    const ConfigParam & configParam, 
    TrainParam & param, 
    Layer layer,    
    std::vector<DFTINFO>& defectInfo)
{
    return 0;
}

int AlgBase::padInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat *img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //采用深度学习方案
    if (configParam.PadParam.usingDL == 1)   return 0;

    //分配内存
    int nMem = 0;
    uchar *knSpc = NULL;
    float *pcsSpc = NULL;
    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }
        lockMemory(gSpaceParam.memoryLock, nMem);
        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = configParam.PadParam.shrinkSize * 2 + 1;
    cv::Mat mask;
    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);
    cv::Mat dst;

    cv::Mat mLower, mUpper;
    cv::Rect dftRect;
    std::vector<cv::Rect> dftRects;
    cv::RotatedRect dftRtRect;
    double dftArea;
    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    bool totalFlag = false;
    double tLower, tUpper;
    cv::Scalar mean;
    //double start;

    //融合轴距离
    int fuseDist = configParam.PadParam.ruseDist;

    cv::Mat channelMat[3];
    cv::Mat subImg;

    const ITEM_REGION *pItem;
    cv::Vec3b *pImg, *pSubimg;
    uchar *pUchar;
    for (int i = 0; i < region.items.size(); i++)
    {

        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        //pItem->mask.copyTo(mask);
        cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        //cv::LUT((*img)(roi), gLut, (*img)(roi));

        //start = cv::getTickCount();
        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);
        channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);
        channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);
        channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);
        subImg = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);
        //hsv = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);
        dst.setTo(0);

        //if (isModify)
        //{
        //    for (int r = 0; r < roi.height; r++)
        //    {
        //        pImg = img->ptr<cv::Vec3b>(r + roi.y);
        //        pSubimg = subImg.ptr<cv::Vec3b>(r);
        //        pUchar = mask.ptr<uchar>(r + maskRoi.y);
        //        for (int c = 0; c < roi.width; c++)
        //        {
        //            if (pUchar[c + maskRoi.x] > 128)
        //            {
        //                pSubimg[c][0] = gLut.ptr(0)[pImg[c + roi.x][0]];
        //                pSubimg[c][1] = gLut.ptr(0)[pImg[c + roi.x][1]];
        //                pSubimg[c][2] = gLut.ptr(0)[pImg[c + roi.x][2]];
        //            }
        //        }
        //    }
        //}

        subImg = (*img)(roi).clone();

        cv::split(subImg, channelMat);

        for (int c = 0; c < 3; c++)
        {
            //整体异色
            cv::meanStdDev(channelMat[c], mean, cv::Scalar::all(0), mask(maskRoi));

            //if (mean.val[0] < gHyperParam.pad[pItem->iID].totalMean[c] - configParam.PadParam.colorParam.lowerOffset[c] ||
            //    mean.val[0] > gHyperParam.pad[pItem->iID].totalMean[c] + configParam.PadParam.colorParam.upperOffset[c])
            //{
            //    dst.setTo(255);
            //    totalFlag = true;
            //    break;
            //}

            //未判断为整体异色的区域先做光照补偿
            //channelMat[c] += gHyperParam.pad[pItem->iID].totalMean[c] - mean.val[0];

            tLower = gHyperParam.pad[pItem->iID].lowerMean[c] - gHyperParam.pad[pItem->iID].lowerStdDev[c] * configParam.PadParam.colorParam.lowerLimit[c];
            tUpper = gHyperParam.pad[pItem->iID].upperMean[c] + gHyperParam.pad[pItem->iID].upperStdDev[c] * configParam.PadParam.colorParam.upperLimit[c];

            cv::threshold(channelMat[c], mLower, tLower, 255, cv::THRESH_BINARY_INV);
            cv::threshold(channelMat[c], mUpper, tUpper, 255, cv::THRESH_BINARY);
            dst = dst + mLower + mUpper;
        }

        ////使用channelMat空间做数据
        //cv::cvtColor((*img)(roi), channelMat[0], cv::COLOR_BGR2GRAY);
        //cv::threshold(channelMat[0], channelMat[0], 0, 255, cv::THRESH_OTSU);
        if (totalFlag == false)
        {
            maskFitting(~dst, mask(maskRoi));
            cv::morphologyEx(mask(maskRoi),
                mask(maskRoi),
                cv::MorphTypes::MORPH_ERODE,
                kernel,
                cv::Point(-1, -1),
                1, cv::BorderTypes::BORDER_CONSTANT, 0);
        }

        dst = dst & mask(maskRoi);
        //std::cout << "pad channles time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;

        //start = cv::getTickCount();

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;

            if (dftArea > configParam.PadParam.colorParam.infArea ||
                dftRtRect.size.width > configParam.PadParam.colorParam.infWidth ||
                dftRtRect.size.height > configParam.PadParam.colorParam.infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back(dftRects[n]);
            }
        }

        dftRects.clear();
        //std::cout << "opacity dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::padInspector_SGM(
	const ABSTRACT_REGIONS & region, 
	const ConfigParam & configParam, 
	cv::Mat * img,
	std::vector<cv::Rect>& defectInfo, 
	bool isModify)
{
	//采用深度学习方案
	if (configParam.PadParam.usingDL == 1)   return 0;

	//分配内存
	int nMem = 0;
	uchar *knSpc = NULL;
	float *pcsSpc = NULL;
	//查找可用内存区域
	{
		std::lock_guard<std::mutex> _lock(muMemMan);
		while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
		{
			Sleep(1);
		}
		lockMemory(gSpaceParam.memoryLock, nMem);
		knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
		pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
	}

	int shrinkSize = configParam.PadParam.shrinkSize * 2 + 1;
	cv::Mat mask;
	cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);
	cv::Mat dst;

	cv::Mat mLower, mUpper;
	cv::Rect dftRect;
	std::vector<cv::Rect> dftRects;
	cv::RotatedRect dftRtRect;
	double dftArea;
	//items的起始结束区域
	int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
	cv::Rect roi, maskRoi;

	bool totalFlag = false;
	double tLower, tUpper;
	cv::Scalar mean;
	//double start;

	//融合轴距离
	int fuseDist = configParam.PadParam.ruseDist;

	cv::Mat channelMat[3];
	cv::Mat subImg;

	const ITEM_REGION *pItem;
	cv::Vec3b *pImg, *pSubimg;
	uchar *pUchar;
	cv::Mat matrixVectorFeature; 
	cv::Mat matrixValueFeature;
	cv::Mat matrixMean;

	cv::Mat imgReshape;
	cv::Mat imgPow;
	cv::Mat imgValue;
	cv::Mat imgDst;
	for (int i = 0; i < region.items.size(); i++)
	{

		pItem = &region.items[i];
		if (pItem->iID < 0)	continue;

		//pItem->mask.copyTo(mask);
		cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

		sx = __max(0, pItem->iOffsetX);
		sy = __max(0, pItem->iOffsetY);
		ex = __min(pItem->iOffsetX + mask.cols, img->cols);
		ey = __min(pItem->iOffsetY + mask.rows, img->rows);
		width = ex - sx;
		height = ey - sy;
		roi = { sx, sy, width, height };
		if (ex <= 0 || width <= 0)	continue;

		maskRoi = { sx - pItem->iOffsetX,
			0,
			width,
			height };

		//cv::LUT((*img)(roi), gLut, (*img)(roi));

		//start = cv::getTickCount();
		dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);
		subImg = cv::Mat(height, width, CV_8UC3, 
			(uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);


		dst.setTo(0);

		if (isModify)
		{
			for (int r = 0; r < roi.height; r++)
			{
				pImg = img->ptr<cv::Vec3b>(r + roi.y);
				pSubimg = subImg.ptr<cv::Vec3b>(r);
				pUchar = mask.ptr<uchar>(r + maskRoi.y);
				for (int c = 0; c < roi.width; c++)
				{
					if (pUchar[c + maskRoi.x] > 128)
					{
						pSubimg[c][0] = gLut.ptr(0)[pImg[c + roi.x][0]];
						pSubimg[c][1] = gLut.ptr(0)[pImg[c + roi.x][1]];
						pSubimg[c][2] = gLut.ptr(0)[pImg[c + roi.x][2]];
					}
				}
			}
		}

		cv::morphologyEx(mask(maskRoi),
			mask(maskRoi),
			cv::MorphTypes::MORPH_ERODE,
			kernel,
			cv::Point(-1, -1),
			1, cv::BorderTypes::BORDER_CONSTANT, 0);

		matrixVectorFeature= (cv::Mat_<float>(3, 3) << 
			gHyperParam.pad[pItem->iID].vectorFeature[0][0],
			gHyperParam.pad[pItem->iID].vectorFeature[0][1], 
			gHyperParam.pad[pItem->iID].vectorFeature[0][2],
			gHyperParam.pad[pItem->iID].vectorFeature[1][0],
			gHyperParam.pad[pItem->iID].vectorFeature[1][1],
			gHyperParam.pad[pItem->iID].vectorFeature[1][2],
			gHyperParam.pad[pItem->iID].vectorFeature[2][0],
			gHyperParam.pad[pItem->iID].vectorFeature[2][1],
			gHyperParam.pad[pItem->iID].vectorFeature[2][2]
			);


		matrixValueFeature = (cv::Mat_<float>(3, 1) <<
			1/gHyperParam.pad[pItem->iID].valueFeature[0]
			/configParam.PadParam.colorParam.lowerLimit[0] 
			/configParam.PadParam.colorParam.lowerLimit[0],
		    1/gHyperParam.pad[pItem->iID].valueFeature[1]
			/configParam.PadParam.colorParam.lowerLimit[1]
			/configParam.PadParam.colorParam.lowerLimit[1],
			1/gHyperParam.pad[pItem->iID].valueFeature[2]  
			/configParam.PadParam.colorParam.lowerLimit[2] 
			/configParam.PadParam.colorParam.lowerLimit[2]
			);


		matrixMean = (cv::Mat_<float>(1, 3) <<
			gHyperParam.pad[pItem->iID].totalMean_SGM[0],
			gHyperParam.pad[pItem->iID].totalMean_SGM[1],
			gHyperParam.pad[pItem->iID].totalMean_SGM[2]
			);
		
		
		subImg.reshape(1, subImg.rows*subImg.cols).convertTo(imgReshape, CV_32FC1);//(subImg.rows*subImg.cols)*3;
		
		imgReshape -= cv::repeat(matrixMean, subImg.rows*subImg.cols, 1);
		cv::pow(imgReshape*matrixVectorFeature.t(),2, imgPow);//N*3
		imgValue = imgPow*matrixValueFeature;
		imgDst = imgValue.reshape(1, subImg.rows);


#if 1
		cv::Mat _matrixVectorFeature = (cv::Mat_<float>(1, 3) <<
			gHyperParam.pad[pItem->iID].vectorFeature[0][0],
			gHyperParam.pad[pItem->iID].vectorFeature[0][1],
			gHyperParam.pad[pItem->iID].vectorFeature[0][2]
			);

		cv::Mat imgTest = imgReshape*_matrixVectorFeature.t();

		cv::Mat _imgReshape = imgTest.reshape(1, subImg.rows);

		cv::Mat _imgAbs = cv::abs(_imgReshape);
		cv::Mat _dst;
		cv::threshold(_imgAbs, _dst, pow(gHyperParam.pad[pItem->iID].valueFeature[0], 0.5)*configParam.PadParam.colorParam.lowerLimit[0], 255, cv::ThresholdTypes::THRESH_BINARY);
		_dst.convertTo(_dst, CV_8UC1);
		_dst &= mask(maskRoi);
#endif
		////float valueThresh = 
		////	gHyperParam.pad[pItem->iID].valueFeature[0]* configParam.PadParam.colorParam.lowerLimit[0] * configParam.PadParam.colorParam.lowerLimit[0]
		////	*gHyperParam.pad[pItem->iID].valueFeature[1] * configParam.PadParam.colorParam.lowerLimit[1] * configParam.PadParam.colorParam.lowerLimit[1]
		////	*gHyperParam.pad[pItem->iID].valueFeature[2] * configParam.PadParam.colorParam.lowerLimit[2] * configParam.PadParam.colorParam.lowerLimit[2];

		////

		cv::threshold(imgDst, dst, 1, 255, cv::ThresholdTypes::THRESH_BINARY);
		dst.convertTo(dst,CV_8UC1);
		dst &= mask(maskRoi);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		for (int c = 0; c < contours.size(); c++)
		{
			dftRtRect = cv::minAreaRect(contours[c]);
			dftRect = cv::boundingRect(contours[c]);
			dftArea = cv::sum(dst(dftRect)).val[0] / 255;

			if (dftArea > configParam.PadParam.colorParam.infArea ||
				dftRtRect.size.width > configParam.PadParam.colorParam.infWidth ||
				dftRtRect.size.height > configParam.PadParam.colorParam.infHeight)
			{
				dftRect.x = sx + dftRect.x - fuseDist;
				dftRect.y = sy + dftRect.y - fuseDist;
				dftRect.width = dftRect.width + fuseDist * 2;
				dftRect.height = dftRect.height + fuseDist * 2;
				//溢出部分融合时修正
				dftRects.push_back(dftRect);
			}
		}

		//融合缺陷区域
		fuseRect(dftRects, fuseDist);

		{
			std::lock_guard<std::mutex> _lock(muRoi);
			for (int n = 0; n < dftRects.size(); n++)
			{
				defectInfo.push_back(dftRects[n]);
			}
		}

		dftRects.clear();
		//std::cout << "opacity dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
	}
	{
		std::lock_guard<std::mutex> _lock(muMemMan);
		unlockMemory(gSpaceParam.memoryLock, nMem);
	}
	return 0;
}

int AlgBase::steelInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //采用深度学习方案
    if (configParam.SteelParam.usingDL == 1)   return 0;

    //分配内存
    int nMem = 0;
    uchar *knSpc = NULL;
    float *pcsSpc = NULL;
    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }
        lockMemory(gSpaceParam.memoryLock, nMem);
        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = configParam.SteelParam.shrinkSize * 2 + 1;

    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);
    cv::Mat dst, dst32f;
    cv::Mat mLower, mUpper, mLower32f, mUpper32f;
    cv::Rect dftRect;
    std::vector<cv::Rect> dftRects;
    cv::RotatedRect dftRtRect;
    float dftArea;
    //cv::Mat labels, centroid, stats;
    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    //整体异色标记
    bool totalFlag = false;

    vec3f tLowers, tUppers, tLower, tUpper;
    cv::Scalar mean;

    double start;
    //融合轴距离
    int fuseDist = configParam.SteelParam.ruseDist;

    cv::Mat channelMat[3];
    cv::Mat subImg;
    cv::Vec3b *pImg, *pSubimg;
    uchar *pUchar;
    //cv::Mat lut = cv::Mat(256,1, CV_8UC1);
    for (int i = 0; i < region.items.size(); i++)
    {
        //std::lock_guard<std::mutex> _lock(muImg);

        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        //pItem->mask.copyTo(mask);
        cv::threshold(pItem->mask, mask, 200, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        start = cv::getTickCount();

        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);
        //opencv3.4.2钢片事先申请内存做split会崩溃
        //channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);
        //channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);
        //channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);
        subImg = cv::Mat(height, width, CV_8UC3, (uchar*)pcsSpc + width * height * 4 * sizeof(uchar), width * 3);
        dst32f = cv::Mat(height, width, CV_32FC1, (uchar*)pcsSpc + width * height * 7 * sizeof(uchar), width * sizeof(float));
        mLower32f = cv::Mat(height, width, CV_32FC1, (uchar*)pcsSpc + width * height * 7 * sizeof(uchar) + width * height * sizeof(float), width * sizeof(float));
        mUpper32f = cv::Mat(height, width, CV_32FC1, (uchar*)pcsSpc + width * height * 7 * sizeof(uchar) + width * height * 2 * sizeof(float), width * sizeof(float));
        dst.setTo(0);
        dst32f.setTo(0);

        subImg = (*img)(roi).clone();
        //if (isModify)
        //{
        //    for (int r = 0; r < roi.height; r++)
        //    {
        //        pImg = img->ptr<cv::Vec3b>(r + roi.y);
        //        pSubimg = subImg.ptr<cv::Vec3b>(r);
        //        pUchar = mask.ptr<uchar>(r + maskRoi.y);
        //        for (int c = 0; c < roi.width; c++)
        //        {
        //            if (pUchar[c + maskRoi.x] > 128)
        //            {
        //                pSubimg[c][0] = gLut.ptr(0)[pImg[c + roi.x][0]];
        //                pSubimg[c][1] = gLut.ptr(0)[pImg[c + roi.x][1]];
        //                pSubimg[c][2] = gLut.ptr(0)[pImg[c + roi.x][2]];
        //            }
        //        }
        //    }
        //}

        cv::split(subImg, channelMat);
        //std::cout<<"sub cols:"<<subImg.cols<<"sub step:"<<subImg.step<<"ch cols:"<<channelMat[0].cols<<"ch step:"<<channelMat[0].step<<std::endl;
#ifndef SHIELD
        
        //std::cout << "steel dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        for (int c = 0; c < 3; c++)
        {
            cv::meanStdDev(channelMat[c], mean, cv::Scalar::all(0), mask(maskRoi));

            //if (mean.val[0] < gHyperParam.steel[pItem->iID].lowerMean[c] - configParam.SteelParam.colorParam.lowerOffset[c] ||
            //    mean.val[0] > gHyperParam.steel[pItem->iID].upperMean[c] + configParam.SteelParam.colorParam.upperOffset[c])
            //{
            //    totalFlag = true;
            //    break;
            //}

            //未判断为整体异色的区域先做光照补偿
            channelMat[c] += gHyperParam.steel[pItem->iID].totalMean[c] - mean.val[0];

            tLowers[c] = gHyperParam.steel[pItem->iID].lowerMean[c] - gHyperParam.steel[pItem->iID].lowerStdDev[c] * configParam.SteelParam.colorParam.lowerLimit[c];
            tUppers[c] = gHyperParam.steel[pItem->iID].upperMean[c] + gHyperParam.steel[pItem->iID].upperStdDev[c] * configParam.SteelParam.colorParam.upperLimit[c];

            cv::threshold(channelMat[c], mLower, tLowers[c], 255, cv::THRESH_BINARY_INV);
            cv::threshold(channelMat[c], mUpper, tUppers[c], 255, cv::THRESH_BINARY);

            //检测区域提取
            dst = dst + mLower + mUpper;

            tLower[c] = __max(1, tLowers[c]);
            tUpper[c] = __min(254, tUppers[c]);

            mLower = tLower[c] - channelMat[c];
            mUpper = channelMat[c] - tUpper[c];

            mLower.convertTo(mLower32f, CV_32FC1);
            mUpper.convertTo(mUpper32f, CV_32FC1);

            //检测面积权重提取（非线性加成）
            cv::pow(mLower32f / tLower[c], configParam.SteelParam.areaWeight, mLower32f);
            cv::pow(mUpper32f / tUpper[c], configParam.SteelParam.areaWeight, mUpper32f);

            dst32f = (cv::max)(dst32f, mUpper32f + mLower32f);
        }
        //std::cout << "steel dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        if (totalFlag == false)
        {
            maskFitting(~dst, mask(maskRoi));
            cv::morphologyEx(mask(maskRoi),
                mask(maskRoi),
                cv::MorphTypes::MORPH_ERODE,
                kernel,
                cv::Point(-1, -1),
                1, cv::BorderTypes::BORDER_CONSTANT, 0);
        }
        else
        {
            defectInfo.push_back(roi);
            continue;
        }

        dst = dst & mask(maskRoi);
        //std::cout << "steel dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        //std::cout << "steel dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);

            dftArea = cv::sum(dst32f(dftRect)).val[0];
            if (dftArea > configParam.SteelParam.colorParam.infArea ||
                dftRtRect.size.width > configParam.SteelParam.colorParam.infWidth ||
                dftRtRect.size.height > configParam.SteelParam.colorParam.infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }
        //std::cout << "steel dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back(dftRects[n]);
            }
        }

        dftRects.clear();
        //std::cout << "opacity dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;

#endif // !SHIELD
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::opacityInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //采用深度学习方案
    if (configParam.OpacityParam.usingDL == 1)   return 0;

    //分配内存
    int nMem = 0;
    uchar *knSpc = NULL;
    uchar *pcsSpc = NULL;
    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }
    int shrinkSize = configParam.OpacityParam.shrinkSize * 2 + 1;

    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);
    cv::Mat dst;
    cv::Mat mLower, mUpper;
    cv::Rect dftRect;
    std::vector<cv::Rect> dftRects;
    cv::RotatedRect dftRtRect;
    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    double tLower[3], tUpper[3];
    int defLeft, defTop, defWidth, defHeight, defArea;
    cv::Scalar mean;

    //融合轴距离
    int fuseDist = configParam.OpacityParam.ruseDist;

    //整体异色标记
    bool totalFlag = false;

    cv::Mat channelMat[3];
    cv::Mat lut[3];
    cv::Mat subImg;

    for (int i = 0; i < 3; i++)
    {
        lut[i] = cv::Mat::zeros(1, 256, CV_8UC1);
        lut[i].setTo(255);
    }
    cv::Vec3b *pVec3b;
    uchar *pUchar;
    int diff;

    double start;

    for (int i = 0; i < region.items.size(); i++)
    {
        //std::lock_guard<std::mutex> _lock(muImg);
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        pItem->mask.copyTo(mask);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        start = cv::getTickCount();

        subImg = (*img)(roi);
        dst = cv::Mat(height, width, CV_8UC1, pcsSpc, width);
        channelMat[0] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * sizeof(uchar), width);
        channelMat[1] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * 2 * sizeof(uchar), width);
        channelMat[2] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * 3 * sizeof(uchar), width);

        //lut[0] = cv::Mat(1, 256, CV_8UC1, pcsSpc + width * height * 4 * sizeof(uchar), 256);
        //lut[1] = cv::Mat(1, 256, CV_8UC1, pcsSpc + width * height * 4 * sizeof(uchar) + 256, 256);
        //lut[2] = cv::Mat(1, 256, CV_8UC1, pcsSpc + width * height * 4 * sizeof(uchar) + 512, 256);
        //lut[0].setTo(255);
        //lut[1].setTo(255);
        //lut[2].setTo(255);
        dst.setTo(0);

        cv::split(subImg, channelMat);
#ifdef _TIME_
        std::cout << "opacity channles time1:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        start = cv::getTickCount();
#endif // _TIME_

        //整体异色
        cv::meanStdDev(subImg, mean, cv::Scalar::all(0), mask(maskRoi));
#ifdef _TIME_
        std::cout << "opacity channles time2:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        start = cv::getTickCount();
#endif // _TIME_

        //      //方案一 119ms
        //      for (int c = 0; c < 3; c++)
        //{		
        //	if (mean.val[c] < gHyperParam.opacity[pItem->iID].lowerMean[c] - configParam.OpacityParam.colorParam.lowerOffset[c] ||
        //		mean.val[c] > gHyperParam.opacity[pItem->iID].upperMean[c] + configParam.OpacityParam.colorParam.upperOffset[c])
        //	{
        //		dst.setTo(255);
        //		totalFlag = true;
        //		break;
        //	}
        //	//未判断为整体异色的区域先做光照补偿
        //	channelMat[c] += gHyperParam.opacity[pItem->iID].totalMean[c] - mean.val[c];
        //	tLower[c] = gHyperParam.opacity[pItem->iID].lowerMean[c] - gHyperParam.opacity[pItem->iID].lowerStdDev[c] * configParam.OpacityParam.colorParam.lowerLimit[c];
        //	tUpper[c] = gHyperParam.opacity[pItem->iID].upperMean[c] + gHyperParam.opacity[pItem->iID].upperStdDev[c] * configParam.OpacityParam.colorParam.upperLimit[c];
        //	cv::threshold(channelMat[c], mLower, tLower[c], 255, cv::THRESH_BINARY_INV);
        //	cv::threshold(channelMat[c], mUpper, tUpper[c], 255, cv::THRESH_BINARY);
        //	dst = dst + mLower + mUpper;
        //}

        //方案二 50ms
        //for (int c = 0; c < 3; c++)
        //{
        //    if (mean.val[c] < gHyperParam.opacity[pItem->iID].lowerMean[c] - configParam.OpacityParam.colorParam.lowerOffset[c] ||
        //        mean.val[c] > gHyperParam.opacity[pItem->iID].upperMean[c] + configParam.OpacityParam.colorParam.upperOffset[c])
        //    {
        //        dst.setTo(255);
        //        totalFlag = true;
        //        break;
        //    }
        //}

#ifdef _TIME_
        std::cout << "opacity channles time3.1:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        start = cv::getTickCount();
#endif // _TIME_

        if (totalFlag == false)
        {
            //未判断为整体异色的区域先做光照补偿
            for (int c = 0; c < 3; c++)
            {
                diff = gHyperParam.opacity[pItem->iID].totalMean[c] - mean.val[c];
                tLower[c] = gHyperParam.opacity[pItem->iID].lowerMean[c] -
                    gHyperParam.opacity[pItem->iID].lowerStdDev[c] *
                    configParam.OpacityParam.colorParam.lowerLimit[c] -
                    diff;
                tUpper[c] = gHyperParam.opacity[pItem->iID].upperMean[c] +
                    gHyperParam.opacity[pItem->iID].upperStdDev[c] *
                    configParam.OpacityParam.colorParam.upperLimit[c] -
                    diff;
                tLower[c] = __max(0, tLower[c]);
                tUpper[c] = __min(255, tUpper[c]);

                lut[c](cv::Rect(tLower[c], 0, tUpper[c] - tLower[c] + 1, 1)) = 0;
                cv::LUT(channelMat[c], lut[c], channelMat[c]);
                dst = dst + channelMat[c];
            }
#ifdef _TIME_
            std::cout << "opacity channles time3.2:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
            start = cv::getTickCount();
#endif // _TIME_
            maskFitting(~dst, mask(maskRoi));
            cv::morphologyEx(mask(maskRoi),
                mask(maskRoi),
                cv::MorphTypes::MORPH_ERODE,
                kernel,
                cv::Point(-1, -1),
                1, cv::BorderTypes::BORDER_CONSTANT, 0);
        }

        ////方案三 550ms
        //for (int c = 0; c < 3; c++)
        //{
        //    if (mean.val[c] < gHyperParam.opacity[pItem->iID].lowerMean[c] - configParam.OpacityParam.colorParam.lowerOffset[c] ||
        //        mean.val[c] > gHyperParam.opacity[pItem->iID].upperMean[c] + configParam.OpacityParam.colorParam.upperOffset[c])
        //    {
        //        dst.setTo(255);
        //        totalFlag = true;
        //        break;
        //    }
        //}
        //if(totalFlag == false)
        //{
        //    for (int c = 0; c < 3; c++)
        //    {
        //        tLower[c] = gHyperParam.opacity[pItem->iID].lowerMean[c] - gHyperParam.opacity[pItem->iID].lowerStdDev[c] * configParam.OpacityParam.colorParam.lowerLimit[c];
        //        tUpper[c] = gHyperParam.opacity[pItem->iID].upperMean[c] + gHyperParam.opacity[pItem->iID].upperStdDev[c] * configParam.OpacityParam.colorParam.upperLimit[c];
        //    }
        //    //未判断为整体异色的区域先做光照补偿
        //    for (int r = 0; r < subImg.rows; r++)
        //    {
        //        pVec3b = subImg.ptr<cv::Vec3b>(r);
        //        pUchar = dst.ptr<uchar>(r);
        //        for (int c = 0; c < subImg.cols; c++)
        //        {
        //            vec3bTemp[0] = pVec3b[c][0] + gHyperParam.opacity[pItem->iID].totalMean[0] - mean.val[0];
        //            vec3bTemp[1] = pVec3b[c][1] + gHyperParam.opacity[pItem->iID].totalMean[1] - mean.val[1];
        //            vec3bTemp[2] = pVec3b[c][2] + gHyperParam.opacity[pItem->iID].totalMean[2] - mean.val[2];
        //            if (vec3bTemp[0] > tLower[0] && vec3bTemp[0] < tUpper[0] &&
        //                vec3bTemp[1] > tLower[1] && vec3bTemp[1] < tUpper[1] &&
        //                vec3bTemp[2] > tLower[2] && vec3bTemp[2] < tUpper[2])
        //                pUchar[c] = 0;
        //            else
        //                pUchar[c] = 1;
        //        }
        //    }
        //}

        dst = dst & mask(maskRoi);
        //剔除细小的缺陷
        cv::morphologyEx(dst, dst,
            cv::MorphTypes::MORPH_OPEN,
            kernel(cv::Rect(0, 0, 3, 3)),
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);

#ifdef _TIME_
        std::cout << "opacity channles time3.3:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        start = cv::getTickCount();
#endif // _TIME_

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

#ifdef _TIME_
        std::cout << "opacity channles time4:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        start = cv::getTickCount();
#endif // _TIME_

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;
            if (dftArea > configParam.OpacityParam.colorParam.infArea ||
                dftRtRect.size.width > configParam.OpacityParam.colorParam.infWidth ||
                dftRtRect.size.height > configParam.OpacityParam.colorParam.infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

#ifdef _TIME_
        std::cout << "opacity channles time5:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        start = cv::getTickCount();
#endif // _TIME_
        //融合缺陷区域
        fuseRect(dftRects, fuseDist);
#ifdef _TIME_
        std::cout << "opacity channles time1:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
#endif // _TIME_

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back(dftRects[n]);
            }
        }

        dftRects.clear();
        //std::cout << "opacity dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::transprencyInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //采用深度学习方案
    if (configParam.TransprencyParam.usingDL == 1)   return 0;

    //分配内存
    int nMem = 0;
    uchar *knSpc = NULL;
    uchar *pcsSpc = NULL;

    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize;

    shrinkSize = configParam.TransprencyParam.shrinkSize * 2 + 1;

    const ITEM_REGION *pItem;

    cv::Mat mask;

    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);

    cv::Mat dstKernel = cv::Mat::ones(cv::Size(3,3), CV_8UC1);

    cv::Mat dst;

    cv::Mat mLower, mUpper, mGray, channelMat[3];

    cv::Rect dftRect;

    std::vector<cv::Rect> dftRects;

    cv::RotatedRect dftRtRect;

    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    uchar *pm, *pr, *pg, *pb;

    cv::Vec3s *pVec3s;

    uchar *pUchar;

    //融合轴距离
    int fuseDist = configParam.TransprencyParam.ruseDist;

    //整体异色标记
    bool totalFlag = false;

    cv::Mat subImg;

    int detectSize = 5;
    //double start;
    for (int i = 0; i < region.items.size(); i++)
    {
        //std::lock_guard<std::mutex> _lock(muImg);

        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        pItem->mask.copyTo(mask);

        cv::threshold(mask, mask, 128, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        cv::morphologyEx(mask(maskRoi),
            mask(maskRoi),
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);

		double area = cv::sum(mask(maskRoi)).val[0] / 255;

		if (area < 50000)   //只取圆
		{
			continue;
		}

        
        dst = cv::Mat(height, width, CV_8UC1, pcsSpc, width);
        //channelMat[0] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * sizeof(uchar), width);
        //channelMat[1] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * 2 * sizeof(uchar), width);
        //channelMat[2] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * 3 * sizeof(uchar), width);
        mGray = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);
        mUpper = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);
        mLower = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);

        subImg = (*img)(roi);

        //subImg.convertTo(img16s, CV_16SC3);
        //cv::GaussianBlur(img16s, blur16s, cv::Size(detectSize, detectSize), 2);
        //img16s = img16s - blur16s;

        //cv::split(img16s, channelMat);
        //
        //for (int k = 0; k < 3; k++)
        //{
        //    )
        //    gHyperParam.transparency[pItem->iID].lower
        //}

        //cv::split(subImg, channelMat);

        //dst = (cv::max)((cv::max)(channelMat[0], channelMat[1]),channelMat[2]);

        cv::cvtColor(subImg, dst, cv::COLOR_RGB2GRAY);

        //cv::blur(dst, mGray, cv::Size(3,3));

        cv::threshold(dst, mUpper, configParam.TransprencyParam.upperTolerance, 1, CV_THRESH_BINARY);

        cv::threshold(dst, mLower, configParam.TransprencyParam.lowerTolerance, 1, CV_THRESH_BINARY_INV);

        dst = mUpper + mLower;

        dst = dst & mask(maskRoi);

        cv::morphologyEx(dst, dst, cv::MorphTypes::MORPH_OPEN, dstKernel, cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, 0);

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0];
            if (dftArea > configParam.TransprencyParam.infArea)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back(dftRects[n]);
            }
        }

        dftRects.clear();
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::nestInspector(const ABSTRACT_REGIONS & region, 
	const ConfigParam & configParam, 
	cv::Mat * img, 
	std::vector<cv::Rect>& defectInfo, 
	bool isModify)
{
    //采用深度学习方案
    if (configParam.TransprencyParam.usingDL == 1)   return 0;

    //分配内存
    int nMem = 0;
    uchar *knSpc = NULL;
    uchar *pcsSpc = NULL;

    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }

        lockMemory(gSpaceParam.memoryLock, nMem);

        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = 15;

    const ITEM_REGION *pItem;

    cv::Mat mask;

    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);

    cv::Mat dstKernel = cv::Mat::ones(cv::Size(3, 3), CV_8UC1);

    cv::Mat dst;

    cv::Mat mLower, mUpper, mGray;

    cv::Rect dftRect;

    std::vector<cv::Rect> dftRects;

    cv::RotatedRect dftRtRect;

    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;

    cv::Rect roi, maskRoi;

    uchar *pm, *pr, *pg, *pb;

    cv::Vec3s *pVec3s;

    uchar *pUchar;

    //融合轴距离
    int fuseDist = configParam.TransprencyParam.ruseDist;

    //整体异色标记
    bool totalFlag = false;

    cv::Mat subImg;

    int detectSize = 5;

    //double start;
    for (int i = 0; i < region.items.size(); i++)
    {
        //std::lock_guard<std::mutex> _lock(muImg);
		totalFlag = false;
        pItem = &region.items[i];

        if (pItem->iID < 0)	continue;

        pItem->mask.copyTo(mask);

        cv::threshold(mask, mask, 128, 255, cv::THRESH_BINARY);

        sx = __max(0, pItem->iOffsetX + 10);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols - 10, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;

        roi = { sx, sy, width, height };

        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        cv::morphologyEx(mask(maskRoi),
            mask(maskRoi),
            cv::MorphTypes::MORPH_ERODE,
            kernel,
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);

        dst = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc, width);
        mGray = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height, width);
        mUpper = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + 2 * width * height, width);
        mLower = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + 3 * width * height, width);

        subImg = (*img)(roi);

        cv::cvtColor(subImg, dst, cv::COLOR_RGB2GRAY);

        cv::blur(dst, mGray, cv::Size(3, 3));

        cv::threshold(mGray, mUpper, 50, 1, CV_THRESH_BINARY);

        cv::threshold(mGray, mLower, 100, 1, CV_THRESH_BINARY_INV);

        dst = mUpper + mLower;

        dst = dst & mask(maskRoi);

        cv::morphologyEx(dst, dst, cv::MorphTypes::MORPH_OPEN, dstKernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;
            if (dftArea > configParam.TransprencyParam.infArea)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back(dftRects[n]);
            }
        }

        dftRects.clear();
        //std::cout << "opacity dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::transprencyInspectorV2(
	const ABSTRACT_REGIONS & region, 
	const ConfigParam & configParam, 
	cv::Mat * img, 
	std::vector<cv::Rect>& defectInfo, 
	bool isModify)
{
    return 0;
}

int AlgBase::lineInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //采用深度学习方案
    if (configParam.LineParam.usingDL == 1)   return 0;

    return 0;
}

int AlgBase::figureInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //采用深度学习方案
    if (configParam.FingerParam.usingDL == 1)   return 0;

    return 0;
}

int AlgBase::holeInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //分配内存
    int nMem = 0;
    uchar *knSpc = NULL;
    float *pcsSpc = NULL;
    //查找可用内存区域
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }
        lockMemory(gSpaceParam.memoryLock, nMem);
        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = (float*)addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel(5, 5, CV_8UC1, knSpc);
    cv::Mat dst;
    cv::Mat mLower, mUpper;
    cv::Rect dftRect;
    std::vector<cv::Rect> dftRects;
    cv::RotatedRect dftRtRect;
    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    double tLower, tUpper;
    int defLeft, defTop, defWidth, defHeight, defArea;
    cv::Scalar mean;

    //融合轴距离
    int fuseDist = configParam.HoleParam.ruseDist;

    cv::Mat channelMat[3];

    //double start;
    for (int i = 0; i < region.items.size(); i++)
    {
        //std::lock_guard<std::mutex> _lock(muImg);

        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        pItem->mask.copyTo(mask);


        //孔洞区域只做未被分割区域
        if (pItem->iOffsetX <= 0 || pItem->iOffsetX + mask.cols >= img->cols - 1) continue;

        //增大孔洞检测范围
        sx = __max(0, pItem->iOffsetX - mask.cols / 4);
        sy = __max(0, pItem->iOffsetY - mask.rows / 4);
        ex = __min(pItem->iOffsetX + 1.5 * mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + 1.5 * mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        dst = cv::Mat(height, width, CV_8UC1, pcsSpc, width);
        channelMat[0] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * sizeof(uchar), width);
        channelMat[1] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 2 * sizeof(uchar), width);
        channelMat[2] = cv::Mat(height, width, CV_8UC1, (uchar*)pcsSpc + width * height * 3 * sizeof(uchar), width);

        cv::split((*img)(roi), channelMat);
        int index = gHyperParam.hole[pItem->iID].chioseChannel;
        //double entropyMin = entropy(channelMat[0]);
        //int indexMin = 0;
        //for (int i = 1; i < 3; i++)
        //{
        //    double t = entropy(channelMat[i]);
        //    if (t < entropyMin)
        //    {
        //        entropyMin = t;
        //        indexMin = i;
        //    }
        //}

        //cv::morphologyEx(mask, mask, cv::MorphTypes::MORPH_DILATE, kernel);
        double meanValue = cv::mean(channelMat[index]).val[0];
        cv::threshold(channelMat[index], dst, meanValue, 255, cv::THRESH_BINARY);

        cv::Point center(dst.cols / 2, dst.rows / 2);
        std::vector<std::vector<cv::Point>> vecContours;
        cv::findContours(dst, vecContours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
        int lengthVecContours = vecContours.size();

        std::vector<std::vector<cv::Point>> vecContoursRing;
        for (int i = 0; i < lengthVecContours; i++)
        {
            int lenghtContours = vecContours[i].size();

            if (lenghtContours < 100)
            {
                continue;
            }

            int flag = cv::pointPolygonTest(vecContours[i], center, false);

            if (flag == 1)
            {
                vecContoursRing.push_back(vecContours[i]);
            }
            else
            {
                continue;
            }
        }
        if (vecContoursRing.size() != 2)
        {
            //圆环断开的话，将输入图像输出；
            defectInfo.push_back({ pItem->iOffsetX,pItem->iOffsetY,mask.cols, mask.rows });
            {
                std::lock_guard<std::mutex> _lock(muMemMan);
                unlockMemory(gSpaceParam.memoryLock, nMem);
            }
            return 0;
        }

        int indexInterRing = 0;
        int indexOuterRing = 1;

        //fit circle of inter contours ;
        double xCenterInterRing, yCenterInterRing, radiusInterRing;
        fitCircle(vecContoursRing[indexInterRing], xCenterInterRing, yCenterInterRing, radiusInterRing);

        //cv::circle((*img)(roi), cv::Point(xCenterInterRing, yCenterInterRing), radiusInterRing, cv::Scalar(255,0,0));
        //cv::drawContours((*img)(roi), vecContoursRing, indexOuterRing, cv::Scalar(0,0,255));

        cv::Point2d centerInterRing(xCenterInterRing, yCenterInterRing);

        int lengthOuterRing = vecContoursRing[indexOuterRing].size();
        int indexDistanceMin = -1;

        float distanceMin = 100000000.f;

        for (int index = 0; index < lengthOuterRing; index++)
        {
            float distance = pow(pow(vecContoursRing[indexOuterRing][index].x - centerInterRing.x, 2) + pow(vecContoursRing[indexOuterRing][index].y - centerInterRing.y, 2), 0.5);

            if (distance < distanceMin)
            {
                distanceMin = distance;
                indexDistanceMin = index;
            }
        }

        double widthMinDistance = distanceMin - radiusInterRing;
        if (widthMinDistance < configParam.HoleParam.offsetParam)
        {
            //圆环口过窄的话，将外圆环的外接矩形输出；
            //defectInfo.push_back(cv::boundingRect(vecContoursRing[indexOuterRing]));
            defectInfo.push_back({ pItem->iOffsetX,pItem->iOffsetY,mask.cols, mask.rows });
            {
                std::lock_guard<std::mutex> _lock(muMemMan);
                unlockMemory(gSpaceParam.memoryLock, nMem);
            }
            return 0;
        }
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::charInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //分配内存
    int nMem = 0;
    uchar *knSpc = NULL;
    uchar *pcsSpc = NULL;
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        //查找可用内存区域
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }
        lockMemory(gSpaceParam.memoryLock, nMem);
        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel(3, 3, CV_8UC1, knSpc);
    cv::Mat dst;
    cv::Mat mLower, mUpper;
    cv::Rect dftRect;
    std::vector<cv::Rect> dftRects;
    cv::RotatedRect dftRtRect;
    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    double tLower, tUpper;
    int defLeft, defTop, defWidth, defHeight, defArea;
    cv::Scalar mean;

    //融合轴距离
    int fuseDist = configParam.CharParam.ruseDist;
    int exLen = 3;

    cv::Mat subImg;
    cv::Mat channelMat[3];

    //double start;
    //取R通道
    for (int i = 0; i < region.items.size(); i++)
    {
        //std::lock_guard<std::mutex> _lock(muImg);
        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        //pItem->mask.copyTo(mask);
        cv::threshold(pItem->mask, mask, 250, 255, cv::THRESH_BINARY);
        mask(cv::Rect(0, 0, 2, mask.rows)) = 0;
        mask(cv::Rect(0, 0, mask.cols, 2)) = 0;
        mask(cv::Rect(mask.cols - 2, 0, 2, mask.rows)) = 0;
        mask(cv::Rect(0, mask.rows - 2, mask.cols, 2)) = 0;

        //cv::morphologyEx(mask, mask, cv::MorphTypes::MORPH_ERODE, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);

        hilditchThin(mask, mask);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        subImg = (*img)(roi);
        cv::split(subImg, channelMat);

        double minEnerge = INT_MAX;
        int index = 0;
        //提取当前最大能量差的分量
        for (int c = 0; c < 3; c++)
        {
            double etp = abs(cv::mean(channelMat[c], mask(maskRoi)).val[0] - cv::mean(channelMat[c], ~mask(maskRoi)).val[0]);
            if (minEnerge < etp)
            {
                minEnerge = etp;
                index = c;
            }
        }

        dst = cv::Mat(height, width, CV_8UC1, pcsSpc, width);
        //cv::cvtColor((*img)(roi), dst, cv::COLOR_BGR2GRAY);

        cv::threshold(channelMat[index], dst, 0, 255, cv::THRESH_OTSU);
        dst = ~dst;
        dst &= mask(maskRoi);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;
            if (dftArea > configParam.CharParam.infArea)
            {
                dftRect.x = sx + dftRect.x - fuseDist - exLen;
                dftRect.y = sy + dftRect.y - fuseDist - exLen;
                dftRect.width = dftRect.width + (fuseDist + exLen) * 2;
                dftRect.height = dftRect.height + (fuseDist + exLen) * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }
        }

        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back(dftRects[n]);
            }
        }

        dftRects.clear();
        //std::cout << "opacity dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

int AlgBase::carveInspector(
    const ABSTRACT_REGIONS & region,
    const ConfigParam & configParam,
    cv::Mat * img,
    std::vector<cv::Rect>& defectInfo,
    bool isModify)
{
    //采用深度学习方案
    if (configParam.CarveParam.usingDL == 1)   return 0;

    int nMem = 0;
    uchar *knSpc = NULL;
    uchar *pcsSpc = NULL;
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        //查找可用内存区域
        while ((nMem = findFreeMemory(gSpaceParam.memoryLock, gSpaceParam.MemorySize)) == -1)
        {
            Sleep(1);
        }
        lockMemory(gSpaceParam.memoryLock, nMem);
        knSpc = addrMemory(gSpaceParam.kernelSpace, gSpaceParam.kernelSpaceTotalWidth, nMem);
        pcsSpc = addrMemory(gSpaceParam.pcsSpace, gSpaceParam.pcsSpaceTotalWidth, nMem);
    }

    int shrinkSize = configParam.CarveParam.shrinkSize * 2 + 1;

    const ITEM_REGION *pItem;
    cv::Mat mask;
    cv::Mat kernel(shrinkSize, shrinkSize, CV_8UC1, knSpc);
    cv::Mat dst;
    cv::Mat mLower, mUpper;
    cv::Rect dftRect;
    std::vector<cv::Rect> dftRects;
    cv::RotatedRect dftRtRect;
    double dftArea;

    //items的起始结束区域
    int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
    cv::Rect roi, maskRoi;

    double tLower[3], tUpper[3];
    int defLeft, defTop, defWidth, defHeight, defArea;
    cv::Scalar mean;

    //融合轴距离
    int fuseDist = configParam.CarveParam.ruseDist;

    //整体异色标记
    bool totalFlag = false;

    cv::Mat channelMat[3];
    cv::Mat lut[3];
    cv::Mat subImg;

    for (int i = 0; i < 3; i++)
    {
        lut[i] = cv::Mat::zeros(1, 256, CV_8UC1);
    }
    cv::Vec3b *pVec3b;
    uchar *pUchar;
    int diff;

    //double start;
    for (int i = 0; i < region.items.size(); i++)
    {
        //std::lock_guard<std::mutex> _lock(muImg);
		totalFlag = false;
		for (int i = 0; i < 3; i++)
		{
			lut[i].setTo(255);
		}

        pItem = &region.items[i];
        if (pItem->iID < 0)	continue;

        pItem->mask.copyTo(mask);

        sx = __max(0, pItem->iOffsetX);
        sy = __max(0, pItem->iOffsetY);
        ex = __min(pItem->iOffsetX + mask.cols, img->cols);
        ey = __min(pItem->iOffsetY + mask.rows, img->rows);
        width = ex - sx;
        height = ey - sy;
        roi = { sx, sy, width, height };
        if (ex <= 0 || width <= 0)	continue;

        maskRoi = { sx - pItem->iOffsetX,
            0,
            width,
            height };

        //start = cv::getTickCount();
        subImg = (*img)(roi);
        dst = cv::Mat(height, width, CV_8UC1, pcsSpc, width);
        channelMat[0] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * sizeof(uchar), width);
        channelMat[1] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * 2 * sizeof(uchar), width);
        channelMat[2] = cv::Mat(height, width, CV_8UC1, pcsSpc + width * height * 3 * sizeof(uchar), width);
        dst.setTo(0);

        cv::split(subImg, channelMat);

        cv::meanStdDev(subImg, mean, cv::Scalar::all(0), mask(maskRoi));

        //for (int c = 0; c < 3; c++)
        //{
        //    if (mean.val[c] < gHyperParam.carve[pItem->iID].lowerMean[c] - configParam.CarveParam.colorParam.lowerOffset[c] ||
        //        mean.val[c] > gHyperParam.carve[pItem->iID].upperMean[c] + configParam.CarveParam.colorParam.upperOffset[c])
        //    {
        //        dst.setTo(255);
        //        totalFlag = true;
        //        break;
        //    }
        //}

        if (totalFlag == false)
        {
            //未判断为整体异色的区域先做光照补偿
            for (int c = 0; c < 3; c++)
            {
                diff = gHyperParam.carve[pItem->iID].totalMean[c] - mean.val[c];
                tLower[c] = gHyperParam.carve[pItem->iID].lowerMean[c] -
                    gHyperParam.carve[pItem->iID].lowerStdDev[c] *
                    configParam.CarveParam.colorParam.lowerLimit[c] -
                    diff;
                tUpper[c] = gHyperParam.carve[pItem->iID].upperMean[c] +
                    gHyperParam.carve[pItem->iID].upperStdDev[c] *
                    configParam.CarveParam.colorParam.upperLimit[c] -
                    diff;
				//std::cout << tLower[c] << " " << tUpper[c] << std::endl;
                tLower[c] = __max(0, tLower[c]);
                tUpper[c] = __min(255, tUpper[c]);

                lut[c](cv::Rect(tLower[c], 0, tUpper[c] - tLower[c] + 1, 1)) = 0;
                cv::LUT(channelMat[c], lut[c], channelMat[c]);
                dst = dst + channelMat[c];
            }

            maskFitting(~dst, mask(maskRoi));
            cv::morphologyEx(mask(maskRoi),
                mask(maskRoi),
                cv::MorphTypes::MORPH_ERODE,
                kernel,
                cv::Point(-1, -1),
                1, cv::BorderTypes::BORDER_CONSTANT, 0);
        }

        dst = dst & mask(maskRoi);
        //剔除细小的缺陷
        cv::morphologyEx(dst, dst,
            cv::MorphTypes::MORPH_OPEN,
            kernel(cv::Rect(0, 0, 3, 3)),
            cv::Point(-1, -1),
            1, cv::BorderTypes::BORDER_CONSTANT, 0);


        //start = cv::getTickCount();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (int c = 0; c < contours.size(); c++)
        {
            dftRtRect = cv::minAreaRect(contours[c]);
            dftRect = cv::boundingRect(contours[c]);
            dftArea = cv::sum(dst(dftRect)).val[0] / 255;
            if (dftArea > configParam.CarveParam.colorParam.infArea ||
                dftRtRect.size.width > configParam.CarveParam.colorParam.infWidth ||
                dftRtRect.size.height > configParam.CarveParam.colorParam.infHeight)
            {
                dftRect.x = sx + dftRect.x - fuseDist;
                dftRect.y = sy + dftRect.y - fuseDist;
                dftRect.width = dftRect.width + fuseDist * 2;
                dftRect.height = dftRect.height + fuseDist * 2;
                //溢出部分融合时修正
                dftRects.push_back(dftRect);
            }

        }
        //融合缺陷区域
        fuseRect(dftRects, fuseDist);

        {
            std::lock_guard<std::mutex> _lock(muRoi);
            for (int n = 0; n < dftRects.size(); n++)
            {
                defectInfo.push_back(dftRects[n]);
            }
        }

        dftRects.clear();
        //std::cout << "opacity dft time:" << (cv::getTickCount() - start) * 1000 / cv::getTickFrequency() << "ms" << std::endl;
    }
    {
        std::lock_guard<std::mutex> _lock(muMemMan);
        unlockMemory(gSpaceParam.memoryLock, nMem);
    }
    return 0;
}

//ORIGINAL VERSION;
#if 0
int AlgBase::itemView(
	std::string savePath,
	const cv::Mat &image,
	ALL_REGION &gItems,
	HyperParam &hyperParam)
{
	int nPcs = gItems.size();
	//items的起始结束区域
	int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
	Layer layer;
	//pcs检测数据的临时变量
	PCS_REGION			*_pcs;
	ABSTRACT_REGIONS	*_abstract;
	ITEM_REGION			*_item;
	cv::Mat msk;
	cv::Rect roi, maskRoi;
	static int num = 0;
	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	for (int n = 0; n < nPcs; n++)
	{
		_pcs = &(gItems[n]);
		//std::cout << "iID:" << _pcs->iID << ":";
		for (int i = 0; i < _pcs->itemsRegions.size(); i++)
		{
			_abstract = &(_pcs->itemsRegions[i]);
			//if (_abstract->type != Layer::lineLay_base)	continue;
			if (_abstract->type != Layer::lineLay_conduct
				&&_abstract->type != Layer::lineLay_nest
				&&_abstract->type != Layer::lineLay_pad)
			{
				continue;
			}

			std::cout << std::to_string(_abstract->type) << ":iID:" << _abstract->iID << std::endl << "{" << std::endl;

			for (int t = 0; t < _abstract->items.size(); t++)
			{
				_item = &(_abstract->items[t]);
				cv::threshold(_item->mask, msk, 100, 255, cv::THRESH_BINARY);
				//msk(cv::Rect(0,0,msk.cols,1)) = 255;
				//msk(cv::Rect(0, msk.rows - 1, msk.cols, 1)) = 255;
				//msk(cv::Rect(0,0,1,msk.rows)) = 255;
				//msk(cv::Rect(msk.cols - 1, 0, 1, msk.rows)) = 255;

				cv::morphologyEx(msk, msk, cv::MorphTypes::MORPH_CLOSE, kernel);
				cv::floodFill(msk, cv::Point(msk.cols / 2, msk.rows / 2), 255);
				cv::morphologyEx(msk, msk, cv::MorphTypes::MORPH_ERODE, kernel, cv::Point(-1, -1), 1, cv::BorderTypes::BORDER_CONSTANT, 0);
				cv::cvtColor(msk, msk, cv::COLOR_GRAY2BGR);

				sx = __max(0, _item->iOffsetX);
				sy = __max(0, _item->iOffsetY);
				ex = __min(_item->iOffsetX + msk.cols, image.cols);
				ey = __min(_item->iOffsetY + msk.rows, image.rows);
				width = ex - sx;
				height = ey - sy;
				roi = { sx, sy, width, height };
				if (ex <= 0 || width <= 0)	continue;

				maskRoi = {
					sx - _item->iOffsetX,
					0,
					width,
					height };

				cv::imwrite("./data/10235/0723/1/" + std::to_string(num) + ".jpg", image(roi)/* & msk(maskRoi)*/);
				cv::imwrite("./data/10235/0723/1/" + std::to_string(num) + "_.jpg", msk(maskRoi));
				cv::imwrite("./data/10235/0723/1/" + std::to_string(num++) + "__.jpg", image(roi) & msk(maskRoi));
				//cv::imwrite("data/result/" + std::to_string(n) + "_" + std::to_string(_abstract->type) + std::to_string(_item->iID) + ".jpg",msk);
				//cv::imwrite("data/result/" + std::to_string(n) + "_" + std::to_string(_abstract->type) + std::to_string(_item->iID) + "_.jpg",_item->mask);
				std::cout << "(iID:" << _item->iID <<
					"iMatchFlag:" << _item->iMatchFlag <<
					",dScalar:" << _item->dScalar <<
					",dTransX:" << _item->dTransX <<
					",dTransY:" << _item->dTransY <<
					",dPhi:" << _item->dPhi <<
					",iOffsetX:" << _item->iOffsetX <<
					",iOffsetY:" << _item->iOffsetY << ")" << std::endl;
			}

			std::cout << "}" << std::endl;
		}
		std::cout << std::endl;
	}
	return 0;
}
#endif

//MODIFY VERSION;2019-07-23;
int AlgBase::itemView(
	std::string savePath,
	const cv::Mat &image,
	ALL_REGION &gItems,
	HyperParam &hyperParam)
{
	int nPcs = gItems.size();
	//items的起始结束区域
	int sx = 0, sy = 0, ex = 0, ey = 0, width = 0, height = 0;
	Layer layer;
	//pcs检测数据的临时变量
	PCS_REGION			*_pcs;
	ABSTRACT_REGIONS	*_abstract;
	ITEM_REGION			*_item;
	cv::Mat msk;
	cv::Rect roi, maskRoi;
	static int num = 0;
	static int ix = 0;
	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	for (int n = 0; n < nPcs; n++)
	{
		int indexPcs = gItems[n].iID;
		_pcs = &(gItems[n]);
		//std::cout << "iID:" << _pcs->iID << ":";
		for (int i = 0; i < _pcs->itemsRegions.size(); i++)
		{
			_abstract = &(_pcs->itemsRegions[i]);
			//if (_abstract->type != Layer::lineLay_base)	continue;
			if (_abstract->type != Layer::pcsContourLay
				/*&&_abstract->type != Layer::lineLay_nest*/
				/*&&_abstract->type != Layer::lineLay_pad*/)
			{
				continue;
			}

			std::cout << std::to_string(_abstract->type) << ":iID:" << _abstract->iID << std::endl << "{" << std::endl;

			for (int t = 0; t < _abstract->items.size(); t++)
			{
				_item = &(_abstract->items[t]);
				cv::threshold(_item->mask, msk, 100, 255, cv::THRESH_BINARY);
				//msk(cv::Rect(0,0,msk.cols,1)) = 255;
				//msk(cv::Rect(0, msk.rows - 1, msk.cols, 1)) = 255;
				//msk(cv::Rect(0,0,1,msk.rows)) = 255;
				//msk(cv::Rect(msk.cols - 1, 0, 1, msk.rows)) = 255;
				
				/*
				cv::morphologyEx(msk, msk, cv::MorphTypes::MORPH_CLOSE, kernel);
				cv::floodFill(msk, cv::Point(msk.cols / 2, msk.rows / 2), 255);
				cv::morphologyEx(msk, msk, cv::MorphTypes::MORPH_ERODE, kernel, cv::Point(-1, -1), 1, cv::BorderTypes::BORDER_CONSTANT, 0);
				cv::cvtColor(msk, msk, cv::COLOR_GRAY2BGR);
				*/
				cv::cvtColor(msk, msk, cv::COLOR_GRAY2BGR);
				sx = __max(0, _item->iOffsetX);
				sy = __max(0, _item->iOffsetY);
				ex = __min(_item->iOffsetX + msk.cols, image.cols);
				ey = __min(_item->iOffsetY + msk.rows, image.rows);
				width = ex - sx;
				height = ey - sy;
				roi = { sx, sy, width, height };
				if (ex <= 0 || width <= 0)	continue;

				maskRoi = {
					sx - _item->iOffsetX,
					0,
					width,
					height };

				int index = _item->iID;
				cv::imwrite("./data/10259/0729/3/" + std::to_string(index)  + "_" + std::to_string(indexPcs) +"_"+ std::to_string(ix) + ".jpg", image(roi)/* & msk(maskRoi)*/);
				cv::imwrite("./data/10259/0729/3/" + std::to_string(index)  + "_" + std::to_string(indexPcs) + "_" + std::to_string(ix) + "_.jpg", msk(maskRoi));
				cv::imwrite("./data/10259/0729/3/" + std::to_string(index)  + "_" + std::to_string(indexPcs) + "_" + std::to_string(ix) + "__.jpg", image(roi) & msk(maskRoi));
				//cv::imwrite("data/result/" + std::to_string(n) + "_" + std::to_string(_abstract->type) + std::to_string(_item->iID) + ".jpg",msk);
				//cv::imwrite("data/result/" + std::to_string(n) + "_" + std::to_string(_abstract->type) + std::to_string(_item->iID) + "_.jpg",_item->mask);
				std::cout << "(iID:" << _item->iID <<
					"iMatchFlag:" << _item->iMatchFlag <<
					",dScalar:" << _item->dScalar <<
					",dTransX:" << _item->dTransX <<
					",dTransY:" << _item->dTransY <<
					",dPhi:" << _item->dPhi <<
					",iOffsetX:" << _item->iOffsetX <<
					",iOffsetY:" << _item->iOffsetY << ")" << std::endl;
			}

			std::cout << "}" << std::endl;
		}
		std::cout << std::endl;
	}
	if (num%2==1)
	{
		ix++;
	}
	num++;
	return 0;
}
void AlgBase::equalHistWithMask(
    cv::Mat src,
    float *normHist,
    cv::Mat dst,
    cv::Mat mask)
{
    if (src.empty() || normHist == nullptr)	return;
    if (dst.empty())	dst.create(src.size(), src.type());
    memset(normHist, 0, sizeof(float) * HIST_SZ);
    int localHistogram[HIST_SZ] = { 0, };
    int globalHistogram_[HIST_SZ] = { 0, };

    mask /= 255;

    const size_t sstep = src.step;

    int width = src.cols;
    int height = src.rows;

    if (src.isContinuous())
    {
        width *= height;
        height = 1;
    }

    double maskArea = width * height - cv::sum(mask).val[0];
    if (!mask.empty())
    {
        for (const uchar* ptr = src.ptr<uchar>(0), *maskPtr = mask.ptr<uchar>(0); height--; ptr += sstep)
        {
            int x = 0;
            for (; x <= width - 4; x += 4)
            {
                int t0 = ptr[x] * maskPtr[x], t1 = ptr[x + 1] * maskPtr[x + 1];
                localHistogram[t0]++; localHistogram[t1]++;
                t0 = ptr[x + 2] * maskPtr[x + 2]; t1 = ptr[x + 3] * maskPtr[x + 3];
                localHistogram[t0]++; localHistogram[t1]++;
            }

            for (; x < width; ++x)
                localHistogram[ptr[x] * maskPtr[x]]++;
        }
        localHistogram[0] -= maskArea;
    }
    else
    {
        for (const uchar* ptr = src.ptr<uchar>(0); height--; ptr += sstep)
        {
            int x = 0;
            for (; x <= width - 4; x += 4)
            {
                int t0 = ptr[x], t1 = ptr[x + 1];
                localHistogram[t0]++; localHistogram[t1]++;
                t0 = ptr[x + 2]; t1 = ptr[x + 3];
                localHistogram[t0]++; localHistogram[t1]++;
            }

            for (; x < width; ++x)
                localHistogram[ptr[x]]++;
        }
    }

    for (int i = 0; i < HIST_SZ; i++)
        for (int j = i; j < HIST_SZ; j++)
            globalHistogram_[j] += localHistogram[i];

    //lut = cv::Mat(1, 256, CV_8UC1);
    //uchar *p = lut.data;
    //for (int i = 0; i < HIST_SZ; i++)
    //{
    //	normHist[i] = localHistogram[i] * 1.f / globalHistogram_[HIST_SZ - 1];
    //	p[i] = uchar((globalHistogram_[i] * 1.f / globalHistogram_[HIST_SZ - 1]) * 255);
    //}

    //cv::LUT(src, lut, dst);
}

float AlgBase::getWeightArea(
    const cv::Mat *channels,
    const cv::Rect &roi,
    const vec3f &tLowers,
    const vec3f &tUppers)
{
    cv::Mat mLowers;
    cv::Mat mUppers;
    cv::Mat dst = cv::Mat::zeros(cv::Size(roi.width, roi.height), CV_32FC1);
    vec3f lower, upper;
    for (int c = 0; c < 3; c++)
    {
        lower[c] = __max(tLowers[c], 1);
        upper[c] = __min(tUppers[c], 254);

        //cv::threshold(channels[c](roi), mLowers, lower[c], 0, cv::THRESH_TOZERO_INV);
        mLowers = lower[c] - channels[c](roi);
        mUppers = channels[c](roi) - upper[c];

        mLowers.convertTo(mLowers, CV_32FC1);
        mUppers.convertTo(mUppers, CV_32FC1);

        mLowers = mLowers / lower[c];
        mUppers = mUppers / (255 - upper[c]);

        dst = (cv::max)(dst, mUppers + mLowers);
    }

    //面积加成
    dst = dst.mul(dst);

    return (float)(cv::sum(dst).val[0]);
}

void AlgBase::maskFitting(
	const cv::Mat & obj, 
	cv::Mat & mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(obj, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Mat _mask;

    int maxArea = 0, area;
    cv::Rect roi, labelRoi;
    for (int n = 0; n < contours.size(); n++)
    {
        roi = cv::boundingRect(contours[n]);
        area = cv::sum(obj(roi)).val[0];
        if (maxArea < area)
        {
            maxArea = area;
            labelRoi = roi;
        }
    }

	if (labelRoi.width == 0 || 
        labelRoi.height == 0 ||
        labelRoi.width < 0.7 * mask.cols ||
        labelRoi.height < 0.7 * mask.rows)
	{
		return;
	}
    
	//偏移问题(只做局部)
	cv::resize(mask, _mask, cv::Size(labelRoi.width, labelRoi.height));
	
    cv::threshold(_mask, _mask, 128, 255, cv::THRESH_BINARY);
    mask.setTo(0);
    _mask.copyTo(mask(labelRoi));
}
